import { useState } from 'react';
import URLForm from '@/components/URLForm';
import PreviewCard from '@/components/PreviewCard';
import LoadingSpinner from '@/components/LoadingSpinner';
import GenerateAd from '@/components/GenerateAd';

type Product = {
  title: string;
  price: string;
  description?: string;
  features: string[];
  images: string[];
  [key: string]: any;
};

export default function Home() {
  const [product, setProduct] = useState<Product | null>(null);
  const [jobId, setJobId] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  return (
    <main className="min-h-screen p-6 bg-gray-100">
      <h1 className="text-3xl font-bold mb-6 text-center">AI Video Ad Generator</h1>
      <URLForm setProduct={setProduct} setLoading={setLoading} />
      {loading && <LoadingSpinner />}
      {product && !loading && (
        <>
          <PreviewCard product={product} />
          <GenerateAd product={product} setJobId={setJobId} />
        </>
      )}
      {jobId && (
        <div className="mt-6 text-center">
          <p className="text-green-600 font-semibold">Ad script generated for job: {jobId}</p>
          <a href={`/preview?job_id=${jobId}`} className="text-blue-500 underline mt-2 inline-block">View Preview</a>
        </div>
      )}
    </main>
  );
}
