# 🚀 Deploying the GNN Web App to the Internet

You now have a fully functional PyTorch Graph Neural Network that produces daily stock picks, and an interactive `app.py` web dashboard to view them!

To share your dashboard with the world (and yourself from your phone), you can deploy it for **free** using Streamlit Community Cloud.

## Deployment Steps (Takes 2 minutes)

1. **Push your code to GitHub (if you haven't already)**
   Make sure `app.py`, `train_script.py`, the `models/` folder, the `data/` folder, `gcn_model.pth`, and `requirements.txt` are pushed to a public or private GitHub repository.

2. **Sign up for Streamlit Community Cloud**
   Go to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.

3. **Deploy the App**
   * Click **"New app"**.
   * Select the GitHub repository where you uploaded this project (e.g., `gnn_supply_chain`).
   * For **Main file path**, type: `app.py`.
   * Click **"Deploy!"**

Streamlit will automatically read your `requirements.txt` file, install PyTorch and the other dependencies in its cloud server, load up your saved AI network weights (`gcn_model.pth`), and instantly give you a live URL predicting today's market! 

*(Optional) You can run the dashboard locally at any time by running `streamlit run app.py` in your terminal!*
