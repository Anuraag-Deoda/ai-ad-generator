// frontend/src/components/URLForm.tsx
import { Dispatch, SetStateAction, useState } from 'react';
import { Form, Button, InputGroup, Spinner } from 'react-bootstrap';

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
  setError: Dispatch<SetStateAction<string>>;
};

export default function URLForm({ setProduct, setLoading, setError }: Props) {
  const [url, setUrl] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  const validateUrl = (url: string): boolean => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    
    if (!url.trim()) {
      setError('Please enter a product URL');
      return;
    }
    
    if (!validateUrl(url)) {
      setError('Please enter a valid URL');
      return;
    }

    setIsSubmitting(true);
    setLoading(true);
    
    try {
      const res = await fetch('http://localhost:5000/api/analyze-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      if (data.success) {
        setProduct(data);
        setUrl(''); // Clear form on success
      } else {
        setError(data.error || 'Failed to analyze product. Please check the URL and try again.');
      }
    } catch (err) {
      console.error('Request failed', err);
      setError('Unable to connect to server. Please check your connection and try again.');
    } finally {
      setLoading(false);
      setIsSubmitting(false);
    }
  };

  return (
    <Form onSubmit={handleSubmit}>
      <InputGroup className="mb-3">
        <Form.Control
          type="url"
          placeholder="Enter product URL (e.g., https://example.com/product)"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          size="lg"
          disabled={isSubmitting}
        />
        <Button 
          variant="primary" 
          type="submit" 
          disabled={isSubmitting || !url.trim()}
          size="lg"
        >
          {isSubmitting ? (
            <>
              <Spinner size="sm" className="me-2" />
              Analyzing...
            </>
          ) : (
            '🔍 Analyze Product'
          )}
        </Button>
      </InputGroup>
    </Form>
  );
}
