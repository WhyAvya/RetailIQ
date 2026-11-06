from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, DateTime, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.db import Base


class UploadMetadata(Base):
    __tablename__ = "upload_metadata"

    upload_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    processing_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="uploaded",
        server_default="uploaded",
    )
