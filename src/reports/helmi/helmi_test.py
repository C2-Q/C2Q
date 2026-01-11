# from src.reports.helmi.problems.maximal_independent_set import MIS
# from src.parser.parser import Parser
#
# import os
# import json
#
# import requests
# # from iqm.iqm_client import IQMClient
# # from iqm.qiskit_iqm import IQMProvider
#
# def get_calibration_data(client: IQMClient, calibration_set_id=None, filename: str = None):
#     """
#     Return the calibration data and figures of merit using IQMClient.
#     Optionally you can input a calibration set id (UUID) to query historical results
#     Optionally save the response to a json file, if filename is provided
#     """
#     headers = {'User-Agent': client._signature}
#     bearer_token = client._get_bearer_token()
#     headers['Authorization'] = bearer_token
#
#     url = os.path.join(client._base_url, 'calibration/metrics/latest')
#     if calibration_set_id:
#         url = os.path.join(url, calibration_set_id)
#
#     response = requests.get(url, headers=headers)
#     response.raise_for_status()  # will raise an HTTPError if the response was not ok
#
#     data = response.json()
#     data_str = json.dumps(data, indent=4)
#
#     if filename:
#         with open(filename, 'w') as f:
#             f.write(data_str)
#         print(f"Data saved to {filename}")
#
#     return data
#
# HELMI_CORTEX_URL = os.getenv('HELMI_CORTEX_URL')  # This is set when loading the module
#
# provider = IQMProvider(HELMI_CORTEX_URL)
# helmi_backend = provider.get_backend()
#
# is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (0, 2), (1, 2), (1, 3)]\nindependent_set = independent_nodes(2, edges)\nprint(independent_set)"
# parser = Parser(model_path="others/saved_models")
#
# problem_type, data = parser.parse(is_snippet)
# data.visualize()
# mis = MIS(data.G)
# mis.report(backend=helmi_backend)
#
# calibration_data = get_calibration_data(helmi_backend.client, filename="calibration.txt")