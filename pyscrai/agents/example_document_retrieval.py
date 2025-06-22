from pyscrai.agents import AgentBuilder, DocumentRetrievalAgent
from pyscrai.config.config import load_template

if __name__ == "__main__":
    # Load config (can be from file or template)
    config = load_template("default")
    # Optionally override storage path or other config here
    agent = DocumentRetrievalAgent()
    repo = "tiangolo/fastapi"
    docs_path = "docs/en/docs"
    tag = ""
    subdir = f"{repo.replace('/', '_')}_{tag}"
    print(f"Fetching documentation from {repo}@{tag}:{docs_path} …")
    files = agent.fetch_github_docs(repo, docs_path, ref=tag, subdir_name=subdir)
    if not files:
        print("No files downloaded.")
    else:
        print(f"Downloaded {len(files)} files:")
        for f in files:
            print(f)
