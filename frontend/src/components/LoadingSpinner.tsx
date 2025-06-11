// frontend/src/components/LoadingSpinner.tsx
export default function LoadingSpinner() {
  return (
    <div className="flex justify-center mt-8">
      <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-500 border-opacity-75"></div>
    </div>
  );
}
