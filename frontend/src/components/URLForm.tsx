// frontend/src/components/URLForm.tsx
import { Dispatch, SetStateAction, useState } from 'react';

type Product = {
  title: string;
  price: string;
  description?: string;
  features: string[];
  images: string[];
  [key: string]: any;
};

type Props = {
  setProduct: Dispatch<SetStateAction<Product | null>>;
  setLoading: Dispatch<SetStateAction<boolean>>;
};

export default function URLForm({ setProduct, setLoading }: Props) {
  const [url, setUrl] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/api/analyze-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      const data = await res.json();
      if (data.success) {
        setProduct(data);
      } else {
        alert(data.error || 'Error analyzing product');
      }
    } catch (err) {
      console.error('Request failed', err);
      alert('Server error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-4 mb-6 justify-center">
      <input
        type="text"
        placeholder="Enter product URL"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        className="w-96 px-4 py-2 border rounded shadow"
      />
      <button type="submit" className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">Analyze</button>
    </form>
  );
}
