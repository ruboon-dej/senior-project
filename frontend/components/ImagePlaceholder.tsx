interface ImagePlaceholderProps {
  label?: string;
  className?: string;
  rounded?: string;
}

/**
 * Marks a spot where a real image/icon/background should go later.
 * Swap this out for an <img> or CSS background-image once assets exist.
 */
export default function ImagePlaceholder({
  label = "Image",
  className = "",
  rounded = "rounded-xl",
}: ImagePlaceholderProps) {
  return (
    <div
      className={`flex items-center justify-center border-2 border-dashed border-gray-300 bg-gray-100 text-gray-400 text-xs text-center px-2 ${rounded} ${className}`}
    >
      🖼&nbsp;{label}
    </div>
  );
}
