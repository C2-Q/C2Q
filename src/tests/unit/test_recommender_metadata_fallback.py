from src.recommender import recommender_engine


def test_select_provider_handles_missing_ibm_metadata_directory(tmp_path, monkeypatch):
    # Simulate a wheel/install where recommender/device_information is absent.
    monkeypatch.setattr(recommender_engine, "dirname", str(tmp_path))
    monkeypatch.setattr(recommender_engine, "DEVICE_INFO_DIR", str(tmp_path / "device_information"))

    device_list, price_list_shots = recommender_engine.select_provider("IBM Quantum", available_devices=[])

    assert device_list == []
    assert price_list_shots == []


def test_select_provider_filters_metadata_backed_devices_without_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(recommender_engine, "dirname", str(tmp_path))
    monkeypatch.setattr(recommender_engine, "DEVICE_INFO_DIR", str(tmp_path / "device_information"))

    device_list, price_list_shots = recommender_engine.select_provider("Amazon Braket", available_devices=[])

    assert "Rigetti Ankaa-2" in device_list
    assert "IQM Garnet" not in device_list
    assert "IonQ Aria (Amazon)" not in device_list
    assert len(device_list) == len(price_list_shots)
