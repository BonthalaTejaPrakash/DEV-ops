# DevOps Intelligence - CI/CD Pipeline Failure Prediction & Optimization

Dense GitLab-style dark UI. Real ML - no fake metrics.

## Quick start
Frontend: cd frontend && npm install && npm run dev
Backend: cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload --port 8000
Full: docker compose up --build

## Demo flow
1. Load Demo Data (Acme Cloud)
2. Failure Prediction -> Use Demo Dataset -> Train Model
3. Enter pre-run features -> Predict
4. Optimization -> Mark Implemented -> Before/After from real runs
5. Pipelines -> Simulate Webhook Run

Security: never put tokens in frontend .env. Tenant isolation by organization_id. Roles Admin/Developer/Viewer.
