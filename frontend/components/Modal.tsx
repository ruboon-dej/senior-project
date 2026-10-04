"use client";

import { ReactNode } from "react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  /** If true, the modal fills the screen instead of a centered card (used for camera capture). */
  fullScreen?: boolean;
}

export default function Modal({
  open,
  onClose,
  children,
  fullScreen = false,
}: ModalProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className={
          fullScreen
            ? "w-full h-full bg-black flex flex-col"
            : "bg-white rounded-2xl shadow-lg max-w-sm w-full mx-4 p-5 relative"
        }
      >
        {children}
      </div>
    </div>
  );
}
