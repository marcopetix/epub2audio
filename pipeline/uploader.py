"""Optional: upload audiobook files to Google Drive."""

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILE = "token.pickle"
CREDENTIALS_FILE = "credentials.json"


def _get_drive_service():
    """Authenticate and return a Google Drive service object."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    token_path = Path(TOKEN_FILE)

    if token_path.exists():
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CREDENTIALS_FILE).exists():
                raise FileNotFoundError(
                    f"{CREDENTIALS_FILE} not found. "
                    "Download it from Google Cloud Console "
                    "(APIs & Services > Credentials > OAuth 2.0 Client ID)"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("drive", "v3", credentials=creds)


def _find_or_create_folder(service, name: str, parent_id: str | None = None) -> str:
    """Find or create a folder on Google Drive. Returns folder ID."""
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(body=metadata, fields="id").execute()
    logger.info(f"Created Drive folder: {name}")
    return folder["id"]


def upload_to_drive(config):
    """Upload all audiobook files to Google Drive."""
    from googleapiclient.http import MediaFileUpload

    service = _get_drive_service()

    # Create folder structure
    root_id = _find_or_create_folder(service, config.gdrive_folder)
    book_id = _find_or_create_folder(service, config.book_title, root_id)

    # Upload all audio files, companions, and metadata
    upload_files = (
        list(config.audio_dir.glob("*.mp3"))
        + list(config.output_dir.glob("*.m4b"))
        + list(config.companions_dir.glob("*.pdf"))
        + list(config.companions_dir.glob("*.html"))
    )

    # Also upload manifest and cover
    manifest = config.output_dir / "manifest.json"
    if manifest.exists():
        upload_files.append(manifest)
    cover = config.output_dir / "cover.png"
    if cover.exists():
        upload_files.append(cover)

    for file_path in upload_files:
        mime_map = {
            ".mp3": "audio/mpeg",
            ".m4b": "audio/mp4",
            ".pdf": "application/pdf",
            ".html": "text/html",
            ".json": "application/json",
            ".png": "image/png",
        }
        mime = mime_map.get(file_path.suffix, "application/octet-stream")

        media = MediaFileUpload(str(file_path), mimetype=mime, resumable=True)
        metadata = {"name": file_path.name, "parents": [book_id]}

        service.files().create(body=metadata, media_body=media, fields="id").execute()
        logger.info(f"Uploaded: {file_path.name}")

    # Get shareable link
    folder_url = f"https://drive.google.com/drive/folders/{book_id}"
    logger.info(f"\nGoogle Drive folder: {folder_url}")
    print(f"\nUploaded to Google Drive: {folder_url}")
