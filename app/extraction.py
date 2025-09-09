import re
import fitz
import docx
from collections import defaultdict


def extract_text_from_pdf(file_byte: bytes) -> str:
    document = fitz.open(stream=file_byte, filetype="pdf")
    return " ".join([page.get_text() for page in document]).strip()


def extract_text_from_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join([p.text for p in document.paragraphs]).strip()


SKILL_ALIASES = {
    # === Programming Languages ===
    "python": ["py"],
    "java": ["core java", "advanced java"],
    "c++": ["cpp"],
    "c": [],
    "javascript": ["js", "vanilla js", "ecmascript"],
    "typescript": ["ts"],
    "r": [],
    "go": ["golang"],
    "ruby": ["ruby on rails", "ror"],
    "scala": [],
    "bash": ["shell scripting", "sh"],
    # === Frontend ===
    "html": ["html5"],
    "css": ["css3"],
    "sass": ["scss"],
    "tailwind": ["tailwind css"],
    "bootstrap": [],
    "javascript frameworks": ["vue", "angular", "react"],
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs"],
    "nextjs": ["next.js"],
    "nuxt": ["nuxtjs", "nuxt.js"],
    "frontend": ["ui development", "client-side", "web design", "ui/ux"],
    # === Backend ===
    "nodejs": ["node", "node.js"],
    "express": ["expressjs", "express.js"],
    "fastapi": [],
    "django": [],
    "flask": [],
    "spring boot": ["spring", "springboot"],
    "graphql": [],
    "api development": ["rest api", "restful api", "web api", "graphql"],
    "backend": ["server-side", "api development", "server programming"],
    # === Databases ===
    "mysql": [],
    "postgresql": ["postgres"],
    "mongodb": ["mongo", "documentdb"],
    "redis": [],
    "sqlite": [],
    "oracle": [],
    "sql": ["structured query language"],
    "nosql": [],
    # === DevOps / Deployment ===
    "docker": [],
    "kubernetes": ["k8s"],
    "ci/cd": ["continuous integration", "continuous deployment"],
    "jenkins": [],
    "nginx": [],
    "apache": [],
    "linux": ["ubuntu", "debian", "centos"],
    "cloud": [
        "aws",
        "gcp",
        "azure",
        "google cloud",
        "amazon web services",
        "microsoft azure",
    ],
    "netlify": [],
    "vercel": [],
    "heroku": [],
    # === ML / AI ===
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "nlp": ["natural language processing", "text processing"],
    "cv": ["computer vision"],
    "supervised learning": [],
    "unsupervised learning": [],
    "reinforcement learning": [],
    "tensorflow": [],
    "keras": [],
    "pytorch": [],
    "scikit-learn": ["sklearn"],
    "xgboost": [],
    "lightgbm": [],
    "mlops": [],
    "transformers": ["huggingface"],
    # === Data Science / Analytics ===
    "data science": ["data analysis", "data analytics"],
    "data engineering": [],
    "data visualization": ["data viz", "visual analytics"],
    "pandas": [],
    "numpy": [],
    "matplotlib": [],
    "seaborn": [],
    "plotly": [],
    "power bi": ["pbi"],
    "tableau": [],
    "excel": ["microsoft excel", "spreadsheets"],
    "statistics": ["statistical analysis", "descriptive stats"],
    "eda": ["exploratory data analysis"],
    # === Other Tools & Libraries ===
    "streamlit": [],
    "jupyter": ["jupyter notebooks", "ipynb"],
    "git": ["version control", "github", "gitlab", "bitbucket"],
    "vscode": ["visual studio code"],
    "postman": [],
    "notebooks": ["jupyter", "colab"],
    "colab": ["google colab"],
    "latex": [],
    "beautifulsoup": ["bs4"],
    "selenium": [],
    "openai": ["gpt", "chatgpt", "llm", "large language models"],
    "huggingface": ["transformers", "hf"],
    # === Soft / Meta Skills ===
    "fullstack": ["full-stack", "full stack", "end-to-end development"],
    "agile": ["scrum", "kanban"],
    "problem solving": ["analytical skills"],
    "communication": ["team collaboration", "interpersonal skills"],
}

ALIAS_MAP = {}
REVERSE_MAP = defaultdict(list)

for skill, aliases in SKILL_ALIASES.items():
    ALIAS_MAP[skill] = skill
    REVERSE_MAP[skill].append(skill)
    for alias in aliases:
        ALIAS_MAP[alias.lower()] = skill
        REVERSE_MAP[skill].append(alias.lower())


def extract_skills(text: str):
    text = text.lower()
    found_skills = {}

    for alias, skill in ALIAS_MAP.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text):
            found_skills[skill] = alias

    return found_skills
