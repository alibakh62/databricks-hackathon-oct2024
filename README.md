# databricks-hackathon-oct2024
This repository contains the codes for the app developed for the Databricks GenAI Hackathon - October 2024

## Setup

```bash
pip install -r requirements.txt
```

## Run
- Set the environment variables in the `.env` file. You can use the `.env.local` as a template.
- Rename `models.yaml.local` to `models.yaml`.
- Run the app with `python app.py`.

## Notes
- The app will generate images in the `generated_images` folder.
- The app will save the fine-tuned model in the GCS bucket. You need to have the `gc_service_account_key.json` file in the same folder as the `app.py` file. You also need to update the `GCS_BUCKET_NAME` in the `.env` file.
- The USERNAME variable in the `.env` file is your username in Replicate.
