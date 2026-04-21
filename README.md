# Data site for the record club

## To run locally:

```
git clone https://github.com/andrewgarmon/record_club.git
pip install -r requirements.txt        # or pip3
streamlit run Records_and_Rebuttals.py
```

## Running the tests

```
pip install -r requirements-dev.txt
pytest
```

Tests also run automatically on every push / pull request via GitHub Actions
(see `.github/workflows/ci.yml`).

## Streamlit Community Cloud Deployment:
http://record-club.streamlit.app
http://records-dev.streamlit.app

To deploy: push to dev, go to the dev deployment and ensure the changes work,
then submit a PR to main.
