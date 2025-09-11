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
    "python": ["py", "python3", "python2"],
    "java": ["core java", "advanced java", "jdk", "jre"],
    "c++": ["cpp", "c plus plus"],
    "c": ["c language", "c programming"],
    "javascript": ["js", "vanilla js", "ecmascript", "es6", "es2015"],
    "typescript": ["ts"],
    "r": ["r language"],
    "go": ["golang"],
    "ruby": ["ruby on rails", "ror"],
    "scala": [],
    "bash": ["shell scripting", "sh", "bash scripting"],
    "kotlin": [],
    "swift": ["ios development", "swiftui"],
    "dart": ["flutter"],
    # === Frontend ===
    "html": ["html5", "hypertext markup language"],
    "css": ["css3", "cascading style sheets"],
    "sass": ["scss", "sassy css"],
    "tailwind": ["tailwind css"],
    "bootstrap": ["bootstrap4", "bootstrap5"],
    "javascript frameworks": ["vue", "angular", "react", "svelte"],
    "react": ["reactjs", "react.js", "react native", "react-native"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs", "angular.js"],
    "nextjs": ["next.js", "next"],
    "nuxt": ["nuxtjs", "nuxt.js"],
    "frontend": ["ui development", "client-side", "web design", "ui/ux", "front-end"],
    # === Backend ===
    "nodejs": ["node", "node.js", "node js"],
    "express": ["expressjs", "express.js"],
    "fastapi": [],
    "django": [],
    "flask": [],
    "spring boot": ["spring", "springboot"],
    "graphql": [],
    "api development": ["rest api", "restful api", "web api", "graphql", "openapi"],
    "backend": ["server-side", "api development", "server programming", "back-end"],
    # === Databases ===
    "mysql": ["my sql"],
    "postgresql": ["postgres", "postgre"],
    "mongodb": ["mongo", "documentdb", "mongo db"],
    "redis": [],
    "sqlite": [],
    "oracle": [],
    "sql": ["structured query language", "tsql", "plsql"],
    "nosql": ["non-relational databases", "non sql"],
    "firebase": ["firestore", "realtime database"],
    "dynamodb": [],
    # === DevOps / Deployment ===
    "docker": [],
    "kubernetes": ["k8s"],
    "ci/cd": ["continuous integration", "continuous deployment", "cicd"],
    "jenkins": [],
    "nginx": [],
    "apache": ["apache server"],
    "linux": ["ubuntu", "debian", "centos", "fedora", "redhat", "arch linux"],
    "cloud": [
        "aws",
        "gcp",
        "azure",
        "google cloud",
        "amazon web services",
        "microsoft azure",
        "cloud computing",
        "cloud services",
    ],
    "netlify": [],
    "vercel": [],
    "heroku": [],
    "terraform": [],
    "ansible": [],
    "gitlab ci": [],
    # === Mobile / App Development ===
    "flutter": ["dart", "flutter framework"],
    "react native": ["react-native", "rn"],
    "android": ["android studio", "kotlin", "android sdk"],
    "ios": ["ios development", "swift", "xcode"],
    # === ML / AI ===
    "machine learning": ["ml", "ml engineer", "ml modeling"],
    "deep learning": ["dl", "neural networks", "cnn", "rnn"],
    "nlp": ["natural language processing", "text processing", "language models"],
    "cv": ["computer vision", "image processing"],
    "supervised learning": [],
    "unsupervised learning": [],
    "reinforcement learning": ["rl"],
    "tensorflow": [],
    "keras": [],
    "pytorch": [],
    "scikit-learn": ["sklearn"],
    "xgboost": [],
    "lightgbm": [],
    "mlops": ["model deployment", "model monitoring"],
    "transformers": ["huggingface", "attention models", "bert", "gpt", "llm"],
    "autoML": ["automl", "h2o.ai", "azure automl"],
    # === Data Science / Analytics ===
    "data science": ["data scientist", "data analytics", "data analysis"],
    "data engineering": ["data pipelines", "etl", "elt"],
    "data visualization": ["data viz", "visual analytics", "dashboards"],
    "pandas": [],
    "numpy": [],
    "matplotlib": [],
    "seaborn": [],
    "plotly": [],
    "power bi": ["pbi", "microsoft power bi"],
    "tableau": [],
    "excel": ["microsoft excel", "spreadsheets", "excel formulas"],
    "statistics": [
        "statistical analysis",
        "descriptive stats",
        "inferential statistics",
    ],
    "eda": ["exploratory data analysis"],
    "big data": ["hadoop", "spark", "pyspark", "hive"],
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
    "openai": ["gpt", "chatgpt", "llm", "large language models", "openai api"],
    "huggingface": ["transformers", "hf"],
    "fasttext": [],
    "nltk": [],
    "spacy": [],
    # === Software Engineering / Meta Skills ===
    "fullstack": [
        "full-stack",
        "full stack",
        "end-to-end development",
        "frontend and backend",
    ],
    "agile": ["scrum", "kanban", "agile methodology"],
    "problem solving": ["analytical skills", "logical reasoning"],
    "communication": [
        "team collaboration",
        "interpersonal skills",
        "verbal communication",
    ],
    "system design": ["high-level design", "low-level design"],
    "object oriented programming": [
        "oop",
        "oop concepts",
        "encapsulation",
        "inheritance",
        "polymorphism",
    ],
    "data structures": [
        "algorithms",
        "dsa",
        "linked list",
        "trees",
        "graphs",
        "sorting",
    ],
    "unit testing": ["pytest", "unittest", "test automation"],
    "software development lifecycle": ["sdlc"],
    # === Consulting / Business / Strategy ===
    "business strategy": [
        "strategic planning",
        "business planning",
        "corporate strategy",
    ],
    "market research": ["industry research", "competitive analysis"],
    "financial modeling": ["excel modeling", "valuation models"],
    "stakeholder management": ["client management", "executive communication"],
    "project management": ["pmp", "project planning", "task tracking"],
    "change management": ["organizational change", "transition planning"],
    "process improvement": ["business process optimization", "bpr", "lean"],
    "problem solving": ["structured thinking", "root cause analysis", "issue trees"],
    "data analysis": ["data-driven decisions", "quantitative analysis"],
    "presentation skills": ["slide decks", "storytelling", "powerpoint"],
    "client communication": [
        "client interaction",
        "presentation",
        "consulting communication",
    ],
    "financial analysis": [
        "ratio analysis",
        "income statement",
        "balance sheet analysis",
    ],
    "swot analysis": ["strengths weaknesses opportunities threats"],
    "benchmarking": ["competitor benchmarking", "industry benchmarks"],
    "go-to-market strategy": ["gtm strategy", "market entry"],
    "business development": ["bd", "sales strategy", "client acquisition"],
    "salesforce": ["crm", "sales software"],
    "crm tools": ["hubspot", "zoho crm", "pipedrive"],
    "microsoft excel": [
        "excel",
        "spreadsheets",
        "excel formulas",
        "vlookup",
        "pivot tables",
    ],
    "powerpoint": ["presentation design", "slide preparation"],
    "consulting frameworks": [
        "bcg matrix",
        "porter's five forces",
        "7s framework",
        "value chain",
    ],
    "management consulting": ["consulting", "strategy consulting"],
    "operations management": ["ops management", "logistics", "supply chain"],
    "product management": ["product strategy", "roadmapping", "feature prioritization"],
    "business intelligence": ["bi tools", "power bi", "tableau", "data dashboards"],
    "growth strategy": ["expansion planning", "growth initiatives"],
    "negotiation": ["deal closing", "contract negotiation", "bargaining"],
    "revenue forecasting": ["sales forecasting", "financial projections"],
    "kpi tracking": ["performance metrics", "dashboarding", "reporting"],
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
