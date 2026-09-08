# Experiment configuration

`evaluation-domains.json` defines the fixed validation and test seeds, continuous
sampling ranges, initialization rules, and geometric acceptance criteria.

```bash
python -m src.generate_eval_scenes --domains configs/evaluation-domains.json \
  --output agent/out/evaluation-scenes-v1 --workers 4
```
