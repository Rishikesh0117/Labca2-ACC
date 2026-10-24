import time
from datetime import datetime, timedelta, timezone
 
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
 
REGION = "ap-south-1"
INSTANCE_ID = "REPLACE_WITH_YOUR_INSTANCE_ID"
cw = boto3.client("cloudwatch", region_name=REGION)
 
 
def make_training_data(n=2500, seed=42):
    # Synthetic labelled examples train the demo classifier.
    # Labels are generated from an example policy, not from production labels.
    df = pd.DataFrame()
    df["cpu_percent"] = pd.Series(dtype=float)
    df["memory_percent"] = pd.Series(dtype=float)
    df["request_count"] = pd.Series(dtype=int)
    rng = __import__("numpy").random.default_rng(seed)
    req = rng.integers(10, 1000, n)
    df["cpu_percent"] = (12 + req / 14 + rng.normal(0, 9, n)).clip(1, 100)
    df["memory_percent"] = (18 + req / 20 + rng.normal(0, 8, n)).clip(1, 100)
    df["request_count"] = req
    score = 0.45 * df.cpu_percent + 0.30 * df.memory_percent + 0.25 * df.request_count / 10
    df["workload_class"] = pd.cut(
        score, bins=[-1, 35, 62, float("inf")],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    return df
 
 
def get_cpu_metric(instance_id):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=10)
    response = cw.get_metric_statistics(
        Namespace="AWS/EC2",
        MetricName="CPUUtilization",
        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
        StartTime=start,
        EndTime=end,
        Period=300,
        Statistics=["Average"]
    )
    points = sorted(response["Datapoints"], key=lambda p: p["Timestamp"])
    if not points:
        raise RuntimeError(
            "No CPU datapoint returned yet. Wait 5-10 minutes after launch and retry."
        )
    return round(float(points[-1]["Average"]), 2)
 
 
def predict_and_recommend(model, cpu, memory, requests):
    row = pd.DataFrame([{
        "cpu_percent": cpu,
        "memory_percent": memory,
        "request_count": requests
    }])
    label = model.predict(row)[0]
    replicas = {"Low": 1, "Medium": 2, "High": 4}[label]
    return label, replicas
 
 
def main():
    df = make_training_data()
    features = ["cpu_percent", "memory_percent", "request_count"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["workload_class"], test_size=0.25,
        random_state=42, stratify=df["workload_class"]
    )
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print("Synthetic holdout accuracy:", round(accuracy_score(y_test, predicted), 4))
    print(classification_report(y_test, predicted))
 
    print("\\nReading live EC2 CPUUtilization from CloudWatch...")
    cpu = get_cpu_metric(INSTANCE_ID)
    print("Latest average EC2 CPUUtilization:", cpu, "%")
    memory = float(input("Enter current memory usage percentage (1-100): "))
    requests = int(input("Enter estimated requests per 5-minute interval: "))
 
    label, replicas = predict_and_recommend(model, cpu, memory, requests)
    if cpu >= 85 or memory >= 90:
        label = "High"
        replicas = 4
        safety = "Triggered: high CPU/memory safety override"
    else:
        safety = "Not triggered"
 
    print("\\n=== CLOUD RESOURCE RECOMMENDATION ===")
    print("Timestamp UTC:", datetime.now(timezone.utc).isoformat())
    print("EC2 instance:", INSTANCE_ID)
    print(f"Inputs: CPU={cpu}%, memory={memory}%, requests={requests}")
    print("Predicted workload:", label)
    print("Recommended simulated replicas:", replicas)
    print("Safety override:", safety)
    print("Note: this demo recommends a replica level; it does not change AWS capacity.")
    with open("recommendation_log.txt", "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()},instance={INSTANCE_ID},"
            f"cpu={cpu},memory={memory},requests={requests},"
            f"class={label},recommended_replicas={replicas},safety={safety}\\n"
        )
    print("Saved recommendation to recommendation_log.txt")
 
 
if __name__ == "__main__":
    main()
