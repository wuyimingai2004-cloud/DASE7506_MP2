# Final submission checklist

Deadline: **31 October 2026, 23:59 (Hong Kong / Beijing time, UTC+8)**.

## Prepared locally

- [x] One selected model: TabM + adaptive Majority-of-the-Bests.
- [x] All inference assets, normalization statistics and configurations.
- [x] Standalone training code and exact commands with pinned dependencies.
- [x] Official validation and public-test score JSON.
- [x] Reproducible validation baselines, feature comparison and aggregation ablation.
- [x] Local CPU timing, asset sizes, seeds, search cost and hashes.
- [x] English report, editable report source, README and reuse/AI disclosure.
- [x] Fixed official data/parser/scorer retained.
- [x] Numerical export and standalone-training reproduction checks.

## Submission links

- Code and report: <https://github.com/wuyimingai2004-cloud/DASE7506_MP2/tree/v1.0.0>
- Complete checkpoint: <https://github.com/wuyimingai2004-cloud/DASE7506_MP2/tree/v1.0.0/checkpoints/tabm_mob>
- Frozen version: `v1.0.0`.
- Official score: **81.79871520342613** (382/467).

## Complete before submitting

- [ ] Review the report, implementation and disclosures; add author/student
      information if desired. Use the correct student ID in the submission form.
- [x] Create a publicly accessible code repository containing this release,
      the report and records. Record its commit/tag as the submitted version.
- [x] Publish a checkpoint link that includes every file in `checkpoints/tabm_mob/`.
      The code repository may include this directory directly.
- [x] Verify the public links can be accessed and the README commands work from
      a clean checkout. Keep the data license and attribution files.
- [ ] Enter **81.79871520342613** as the score, with matching code and checkpoint
      links, at <https://xudongwu-0.github.io/courses/dase7506/mp2/>.
- [ ] Click **Submit on GitHub** and create the generated issue. Preparing the
      website link alone does not submit the result.

The course reference CPU determines timing eligibility. Local timing does not
certify performance on that machine. Publishing the repository does not submit
a course result; create the submission issue separately.
