# Mixed-credential composite correction

**Scope.** `switchyard-runner` composition, route execution and the LLM client's candidate
loop. No manifest bytes change; this is the code that decides *whether* a child may be
called with the candidates it has left.

## The defect

A composite recorded a credential block and then judged it against the child's **first**
target. That is only correct when every target in the child shares one credential — which
was true when the composite was written and is false for a free pool that spans providers.

Two visible failures followed:

1. **A healthy peer was never asked.** An OpenRouter free-scope exhaustion recorded against
   the `or` credential skipped an entire child whose first candidate was an OpenRouter leaf,
   even though the child also held a Gemini leaf on a different account.
2. **A per-model Google budget became terminal.** Google's
   `generate_content_free_tier_requests` is allocated **per model inside the project**, and
   the 429 it returns for that carries no block at all in the old code. A composite with
   `cross_on = "credential_block"` treats "no block" as terminal, so one spent model ended a
   request that had six live siblings.

A third, latent failure mattered once the second was fixed: a block for one model's daily
budget would have become a *reason to spend the paid leg*, which the owner's policy does not
authorize.

## The correction

A credential is a property of **the target that failed**, not of the route.

| change | effect |
|---|---|
| `ClientRouting::target_credentials` + `credential_for()` | the router can name the credential of the candidate it is about to try, so "different credential" is a fact rather than an inference |
| `CredentialBlockKind::ModelQuota` | a per-model Google budget retires exactly that target; `WholeQuota` is reserved for a scope Google positively stated as project-wide |
| `CredentialBlock::crosses_to_paid()` | only a whole-credential scope (free allowance, balance, rejected key) authorizes the paid leg |
| `Route::servable_after_blocks()` / `target_is_blocked()` | blocks are judged per target, on that target's credential |
| `Route::execute_with_blocks()` | survivors are executed with the blocks still applied, so a later child cannot spend a call on a target already known spent |
| composition skip condition | a child is skipped only when **none** of its targets remain servable |
| `call_first_available` credential-scoped continuation | a failure that ends its own credential does not end a candidate on another |
| `credential_scoped_stop()` | 401, account-wide quota, or a Google stop that does not keep the outcome — an outcome about the *request* is never evaded by another provider |

## Tests

`crates/switchyard-server/tests/mixed_credential.rs`, seven tests, each naming the
observation that rules the alternative out:

| test | what it rules out |
|---|---|
| `a_spent_free_allowance_on_one_provider_leaves_the_other_servable` | treating an account-wide cap as terminal for the route |
| `a_google_project_quota_leaves_the_openrouter_peer_servable` | a fix that only reads OpenRouter's spelling of the same fact |
| `a_model_scoped_google_quota_retires_only_that_model` | reading a per-model budget as project-wide |
| `a_model_scoped_quota_does_not_spend_the_paid_leg` | buying a paid call to work around one model's daily allocation |
| `a_block_on_one_provider_does_not_skip_a_mixed_child` | judging a block against the child's first target only |
| `both_free_credentials_exhausted_crosses_to_the_paid_leg` | satisfying "never spend" by refusing everything |
| `a_policy_refusal_is_not_evaded_by_another_provider` | re-sending a refused payload to a second provider |

## Mutation controls

Each mutation reverts one half of the correction. All six are caught **by the test that
states that property**, and the tree is green again after restore.

| mutation | caught by |
|---|---|
| child skip judged on first target only | `a_block_on_one_provider_does_not_skip_a_mixed_child` |
| no cross-credential fallback | `a_spent_free_allowance_…`, `a_google_project_quota_…` |
| policy refusal treated as credential scope | `a_policy_refusal_is_not_evaded_by_another_provider` |
| `ModelQuota` crosses to paid | `a_model_scoped_quota_does_not_spend_the_paid_leg` |
| model scope promoted to `WholeQuota` | `a_model_scoped_quota_does_not_spend_the_paid_leg` |
| blocks not passed to the child | `a_block_on_one_provider_does_not_skip_a_mixed_child` |

The two tests that are caught by two different mutations are not redundant: one mutation
removes the *fallback*, the other removes the *filter*, and they fail at different call
counts.

## What this deliberately does not change

* A block on one model of a credential still stops **that credential's** remaining
  candidates; the cross-credential continuation requires a genuinely different credential.
* `cross_on = "failure_class"` composites keep their historical rule; only the
  `credential_block` policy gains the scope distinction.
* No privacy gate, classifier, origin attestation or payload clearance is added. The ST\*
  privacy policy remains deferred by owner decision (charter §0D) until after the first live
  deployment.
