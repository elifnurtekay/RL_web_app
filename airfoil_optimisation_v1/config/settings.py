from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'dev-only-change-me'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'optimizer',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {'context_processors': [
            'django.template.context_processors.request',
            'django.contrib.auth.context_processors.auth',
            'django.contrib.messages.context_processors.messages',
        ]},
    }
]

WSGI_APPLICATION = 'config.wsgi.application'
ASGI_APPLICATION = 'config.asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'optimizer' / 'static']
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ============================================================
# Real PPO / TD3 / SAC model inference settings
# ============================================================

MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

RL_MODEL_ARTIFACTS = {
    "PPO": {
        "model_path": MODEL_ARTIFACTS_DIR / "models" / "ppo_model.zip",
        "metadata_path": MODEL_ARTIFACTS_DIR / "models" / "ppo_metadata.json",
    },
    "TD3": {
        "model_path": MODEL_ARTIFACTS_DIR / "models" / "td3_model.zip",
        "metadata_path": MODEL_ARTIFACTS_DIR / "models" / "td3_metadata.json",
    },
    "SAC": {
        "model_path": MODEL_ARTIFACTS_DIR / "models" / "sac_model.zip",
        "metadata_path": MODEL_ARTIFACTS_DIR / "models" / "sac_metadata.json",
    },
}

RL_SURROGATE_MODEL_NAME = "S-3D"
RL_SURROGATE_CHECKPOINT_PATH = MODEL_ARTIFACTS_DIR / "surrogate" / "surrogate_s3d.pt"
RL_SCALER_JSON_PATH = MODEL_ARTIFACTS_DIR / "surrogate" / "scalers.json"

RL_WEB_EVALUATOR = "surrogate"

