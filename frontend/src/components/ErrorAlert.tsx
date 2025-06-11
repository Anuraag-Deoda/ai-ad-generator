type Props = {
  message: string;
  onClose: () => void;
};

export default function ErrorAlert({ message, onClose }: Props) {
  return (
    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative max-w-xl mx-auto mb-4">
      <strong className="font-bold">Error: </strong>
      <span>{message}</span>
      <span onClick={onClose} className="absolute top-0 bottom-0 right-0 px-4 py-3 cursor-pointer">
        ×
      </span>
    </div>
  );
}
