import yaml
import subprocess

# Read and parse the YAML file
config_file = 'configs/model_config_dev.yml'
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Retrieve the list of ollama_hosts
ollama_hosts = config.get("ollama_hosts", [])
print(ollama_hosts)

# Function to check if a model exists and download it if not
def ensure_model_exists(model_name):
    try:
        # List existing models to check if the model is already available
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        available_models = result.stdout
        if model_name not in available_models:
            print(f"Model '{model_name}' not found. Downloading...")
            pull_result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
            if pull_result.returncode == 0:
                print(f"Model '{model_name}' downloaded successfully.")
            else:
                print(f"Failed to download model '{model_name}': {pull_result.stderr}")
        else:
            print(f"Model '{model_name}' is already available.")
    except Exception as e:
        print(f"Error checking or downloading the model '{model_name}': {e}")

# Loop through each model and size configuration
for model in ollama_hosts:
    model_sizes = ollama_hosts[model]
    model_names = [f"{model}:{size}" for size in model_sizes]
    print(model_names)
    for model_name in model_names:
        ensure_model_exists(model_name)
