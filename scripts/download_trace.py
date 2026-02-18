import os
import requests

from llm_spice.utils.common import PROJECT_ROOT



def download_file(url, folder):
    os.makedirs(folder, exist_ok=True)
    local_filename = url.rsplit('/', 1)[-1]
    path = os.path.join(folder, local_filename)

    print(f"Downloading {local_filename} from {url} to {folder}")
    r = requests.get(url, timeout=60)  # no stream=True
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)  # auto-decompressed by Requests
    return path

def download_azure_trace():
    folder = f"{PROJECT_ROOT}/traces/azure"
    os.makedirs(folder, exist_ok=True)
    download_file(f"https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv", folder)
    download_file(f"https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_conv.csv", folder)
    download_file(f"https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv", folder)
    download_file(f"https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv", folder)

def download_all_trace():
    download_azure_trace()

if __name__ == "__main__":
    download_all_trace()

