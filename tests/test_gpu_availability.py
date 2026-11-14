"""Verify GPU candidate inspection without AWS credentials or billable calls."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load_module():
    path=ROOT/"scripts"/"27_check_sagemaker_gpu.py"
    spec=importlib.util.spec_from_file_location("gpu_inspection",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePaginator:
    def __init__(self,pages):
        self.pages=pages

    def paginate(self,**kwargs):
        return iter(self.pages)


class FakeQuotaClient:
    def get_paginator(self,name):
        assert name=="list_service_quotas"
        return FakePaginator([{"Quotas":[
            {"QuotaName":"ml.g6.2xlarge for training job usage","Value":2.0},
            {"QuotaName":"ml.g6e.2xlarge for training job usage","Value":0.0},
            {"QuotaName":"ml.g5.2xlarge for spot training job usage","Value":5.0},
            {"QuotaName":"ml.g5.2xlarge for training job usage","Value":1.0},
        ]}])


class FakeSageMaker:
    def get_paginator(self,name):
        assert name=="list_training_jobs"
        return FakeStatusPaginator()

    def describe_training_job(self,TrainingJobName):
        assert TrainingJobName=="occupied-g5"
        return {"ResourceConfig":{"InstanceType":"ml.g5.2xlarge","InstanceCount":1}}


class FakeStatusPaginator:
    def paginate(self,StatusEquals):
        if StatusEquals=="InProgress":
            return iter([{"TrainingJobSummaries":[{"TrainingJobName":"occupied-g5"}]}])
        assert StatusEquals=="Stopping"
        return iter([{"TrainingJobSummaries":[]}])


def test_gpu_quota_and_active_job_estimation():
    m=load_module()
    quotas=m.list_quotas(FakeQuotaClient())
    assert quotas["ml.g6.2xlarge"]==2.0
    assert quotas["ml.g6e.2xlarge"]==0.0
    used,jobs=m.list_running(FakeSageMaker())
    assert used["ml.g5.2xlarge"]==1
    assert len(jobs)==1
    rows={x["instance"]:x for x in m.inspect(quotas,used)}
    assert rows["ml.g5.2xlarge"]["status"]=="QUOTA IN USE"
    assert rows["ml.g6.2xlarge"]["estimated_remaining_quota"]==2.0
    assert rows["ml.g6e.2xlarge"]["status"]=="QUOTA ZERO"
    assert rows["ml.g5.4xlarge"]["status"]=="QUOTA UNKNOWN"
