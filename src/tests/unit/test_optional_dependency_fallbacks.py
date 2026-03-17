import pickle

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from src.algorithms.QAOA import QAOA
from src.recommender import recommender_engine


def test_select_device_ionq_uses_generic_backend_when_sdk_missing(tmp_path, monkeypatch):
    device_info_dir = tmp_path / "device_information"
    device_info_dir.mkdir()
    payload = {
        "errors": {"single_qubit": 0.001, "two_qubit": 0.02, "measurement": 0.005},
        "gate_timings": {"single_qubit": 2.0e-5, "two_qubit": 2.0e-4},
        "relaxation_times": {"t1": 10.0, "t2": 1.0},
    }
    with open(device_info_dir / "IonQ Aria (Amazon).pkl", "wb") as handle:
        pickle.dump(payload, handle)

    monkeypatch.setattr(recommender_engine, "DEVICE_INFO_DIR", str(device_info_dir))
    monkeypatch.setattr(recommender_engine, "IonQProvider", None)
    monkeypatch.setattr(recommender_engine, "_IONQ_PROVIDER", None)

    device_dict = recommender_engine.select_device("IonQ Aria (Amazon)", ibm_service=None, available_devices=[])

    assert isinstance(device_dict["backend"], GenericBackendV2)
    assert device_dict["device_qubits"] == 25


def test_select_device_quantinuum_uses_generic_backend_when_extension_missing(monkeypatch):
    monkeypatch.setattr(recommender_engine, "QuantinuumBackend", None)
    monkeypatch.setattr(recommender_engine, "qiskit_to_tk", None)
    monkeypatch.setattr(recommender_engine, "tk_to_qiskit", None)

    device_dict = recommender_engine.select_device("Quantinuum H1", ibm_service=None, available_devices=[])

    assert isinstance(device_dict["backend"], GenericBackendV2)
    assert device_dict["device_qubits"] == 20


def test_qaoa_sampler_falls_back_to_aer_when_runtime_missing(monkeypatch):
    def _raise_import_error():
        raise ImportError("runtime unavailable")

    monkeypatch.setattr(QAOA, "_load_runtime_primitives", _raise_import_error)

    estimator_kind, _ = QAOA._create_estimator(AerSimulator())
    sampler_kind, _ = QAOA._create_sampler(AerSimulator())

    qaoa_dict = QAOA.qaoa_no_optimization(np.array([[-1.0]]), layers=1)
    result = QAOA.sample_results(
        qaoa_dict["qc"], qaoa_dict["parameters"], qaoa_dict["theta"], backend=AerSimulator()
    )

    assert estimator_kind == "aer_v1"
    assert sampler_kind == "aer_v1"
    assert result.shape == (1,)
