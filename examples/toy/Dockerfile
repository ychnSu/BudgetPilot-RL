FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && pip install pytest
WORKDIR /repo
COPY repo/ /repo/
RUN git init && git -c user.name=BudgetPilot -c user.email=demo@example.invalid add . && git -c user.name=BudgetPilot -c user.email=demo@example.invalid commit -m "Toy base checkout"
