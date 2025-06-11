
type Product = {
  title: string;
  price: string;
  description?: string;
  features: string[];
  images: string[];
};

type Props = {
  product: Product;
};

export default function PreviewCard({ product }: Props) {
  return (
    <div className="max-w-xl mx-auto bg-white shadow rounded-lg p-6 mb-6">
      <img src={product.images?.[0]} alt={product.title} className="w-full h-auto object-contain mb-4 rounded" />
      <h2 className="text-2xl font-bold mb-2">{product.title}</h2>
      <p className="text-lg font-medium text-blue-600 mb-2">{product.price}</p>
      {product.description && <p className="text-gray-700 mb-4">{product.description}</p>}
      <ul className="list-disc list-inside text-gray-800">
        {product.features.slice(0, 5).map((feature, idx) => (
          <li key={idx}>{feature}</li>
        ))}
      </ul>
    </div>
  );
}
