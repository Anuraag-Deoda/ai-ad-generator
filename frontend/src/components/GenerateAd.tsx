// frontend/src/components/GenerateAd.tsx
import { Dispatch, SetStateAction, useState } from 'react';

type Product = {
  title: string;
  price: string;
  features: string[];
  description?: string;
  images: string[];
};

type Props = {
  product: Product;
  setJobId: Dispatch<SetStateAction<string>>;
};

export default function GenerateAd({ product, setJobId }: Props) {
  const [generating, setGenerating] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const res = await fetch('http://localhost:5000/api/generate-content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: `job_${Date.now()}`,
          product,
          style: 'energetic',
          duration: 30,
          include_metadata: true,
          include_variations: false
        })
      });

      const data = await res.json();
      if (data.success && data.job_id) {
        setJobId(data.job_id);
      } else {
        alert(data.error || 'Generation failed');
      }
    } catch (err) {
      console.error(err);
      alert('Server error');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="text-center mt-4">
      <button
        onClick={handleGenerate}
        className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700"
        disabled={generating}
      >
        {generating ? 'Generating...' : 'Generate Ad'}
      </button>
    </div>
  );
}
