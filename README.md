# CM4AI_LLM_Challenge

1. Create a W&B account (https://wandb.ai)
2. Connect to compute environment
3. Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
4. Create a new conda environment
```bash
conda create -n llm_challenge python=3.9
```
5. Activate environment and install dependencies
```bash
conda activate llm_challenge
```
6. Clone the CM4AI_LLM_Challenge repository and install dependencies
```
git clone https://github.com/idekerlab/CM4AI_LLM_Challenge.git
cd CM4AI_LLM_Challenge
pip install -r requirements.txt
```
7. Login to W&B with your API key
```bash
wandb login
```
8. Run template code and visualize results in W&B
```bash
python src/evaluate.py
```
