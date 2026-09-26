# Regulatory ruleset note

The built-in Step 4 rules are a **screening/demo subset** for REACH SVHC, RoHS restricted substances and ELV restricted substances. They are intentionally conservative: a CAS not present in the configured subset is shown as `UNKNOWN`, not `PASS`.

Before production use, replace/validate these tables against the current applicable legal datasets and business scope, including:
- REACH Candidate List / applicable REACH obligations
- RoHS Directive and applicable exemptions/scope
- ELV Directive and applicable exemptions/scope
- Jurisdiction, article/product scope and substance form

A CAS-number + concentration comparison alone is not a legal compliance determination.
