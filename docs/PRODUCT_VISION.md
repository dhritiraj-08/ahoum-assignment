# Product Vision

## What this could become

A tool giving professionals a fast, evidence-grounded read on someone's
communication patterns from a conversation transcript -- a therapist
glancing at it before a session, a coach tracking how a client's language
shifts over months. Not a personality test, not a diagnosis: a system
that only says what it can point to in the text, and stays silent when it
can't.

## Who this is for

**Therapists** and **executive coaches**, as a prep or pattern-tracking
aid, never diagnostic. **Researchers** doing qualitative interview
analysis at volume. **HR teams** -- listed deliberately, as the group I
trust least: onboarding/team workshops are legitimate, but hiring
screening is one policy decision away, and nothing technical stops it.

## Ethical considerations and risks

- **Hiring discrimination**: a 1-5 "Assertiveness" score on a candidate is
  a discrimination claim waiting to happen.
- **Snippet, not assessment**: a few sentences isn't a clinical interview;
  any score implies more confidence than the evidence supports.
- **Cultural/linguistic bias**: llama3.1 wasn't trained evenly across
  cultures and dialects -- code-switched speech meets an uneven prior
  about what "confidence" sounds like.
- **GDPR**: personality inferences are plausibly special-category data
  under Article 9; storing them triggers erasure obligations this system
  doesn't handle today.

## Safeguards before production

- A qualified professional reviews any `sensitivity: high` score before
  it reaches a decision -- never surfaced raw.
- Explicit, informed consent before a conversation is scored at all.
- A right to see the evidence behind a score and to contest or remove it.
- Recurring bias audits across demographic/linguistic groups, published.

## What makes this architecture ethical by design

**Abstention is the default**, not a fallback -- the prompt and the
two-layer filter both assume "say nothing" is correct until proven
otherwise. **Sensitivity is a first-class catalogue column**, not a
runtime guess. **Every score ships with the evidence sentence that
produced it**, so a human can check the model's reasoning against the
transcript. **The local Ollama path means the conversation never has to
leave the machine it was typed on.**

## What I find interesting about this problem

What's stuck with me isn't the scoring -- it's that the real engineering
problem is teaching a system to *not* answer. Every instinct in how LLMs
get built pushes toward "always produce a helpful response." Here, the
correct behavior most of the time is silence. Getting a model to abstain
reliably, and a pipeline to structurally block it from even being asked
the wrong question, was a more interesting problem than the scoring itself.
