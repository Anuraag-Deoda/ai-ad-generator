
import { useState } from 'react';
import { Card, Badge, ListGroup, Image, Placeholder } from 'react-bootstrap';

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
  const [imageLoading, setImageLoading] = useState(true);
  const [imageError, setImageError] = useState(false);

  const handleImageLoad = () => setImageLoading(false);
  const handleImageError = () => {
    setImageLoading(false);
    setImageError(true);
  };

  return (
    <Card className="h-100 shadow-sm">
      <div className="position-relative">
        {imageLoading && (
          <Placeholder as="div" animation="glow" className="w-100" style={{ height: '250px' }}>
            <Placeholder xs={12} style={{ height: '100%' }} />
          </Placeholder>
        )}
        {!imageError && product.images?.[0] ? (
          <Image
            src={product.images[0]}
            alt={product.title}
            className={`card-img-top ${imageLoading ? 'd-none' : ''}`}
            style={{ height: '250px', objectFit: 'cover' }}
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        ) : (
          !imageLoading && (
            <div 
              className="card-img-top d-flex align-items-center justify-content-center bg-light text-muted"
              style={{ height: '250px' }}
            >
              <div className="text-center">
                <i className="bi bi-image fs-1"></i>
                <p className="mb-0">No image available</p>
              </div>
            </div>
          )
        )}
      </div>
      
      <Card.Body>
        <Card.Title className="h4 mb-3">{product.title}</Card.Title>
        <div className="mb-3">
          <Badge bg="success" className="fs-6 px-3 py-2">{product.price}</Badge>
        </div>
        
        {product.description && (
          <Card.Text className="text-muted mb-3">
            {product.description}
          </Card.Text>
        )}
        
        {product.features && product.features.length > 0 && (
          <>
            <h6 className="fw-bold mb-2">Key Features:</h6>
            <ListGroup variant="flush">
              {product.features.slice(0, 5).map((feature, idx) => (
                <ListGroup.Item key={idx} className="px-0 py-1">
                  <i className="bi bi-check-circle-fill text-success me-2"></i>
                  {feature}
                </ListGroup.Item>
              ))}
              {product.features.length > 5 && (
                <ListGroup.Item className="px-0 py-1 text-muted fst-italic">
                  +{product.features.length - 5} more features...
                </ListGroup.Item>
              )}
            </ListGroup>
          </>
        )}
      </Card.Body>
    </Card>
  );
}
