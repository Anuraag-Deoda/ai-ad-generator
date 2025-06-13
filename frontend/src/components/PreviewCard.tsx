import { useState } from 'react';
import { Card, Badge, ListGroup, Image, Placeholder, Row, Col } from 'react-bootstrap';
import { motion } from 'framer-motion';

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
    <>
      <style jsx>{`
        .preview-card {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          overflow: hidden;
          transition: all 0.3s ease;
          height: 100%;
        }
        
        .preview-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        }
        
        .card-header-custom {
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border-bottom: 1px solid #e9ecef;
          padding: 16px 20px;
        }
        
        .header-title {
          font-size: 1.1rem;
          font-weight: 600;
          color: #212529;
          margin: 0;
          display: flex;
          align-items: center;
        }
        
        .header-icon {
          margin-right: 8px;
          color: #0d6efd;
        }
        
        .product-image-container {
          position: relative;
          height: 200px;
          overflow: hidden;
          background: #f8f9fa;
        }
        
        .product-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: all 0.3s ease;
        }
        
        .preview-card:hover .product-image {
          transform: scale(1.05);
        }
        
        .image-placeholder {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100%;
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          color: #6c757d;
          flex-direction: column;
        }
        
        .placeholder-icon {
          font-size: 2rem;
          margin-bottom: 8px;
        }
        
        .product-title {
          font-size: 1.2rem;
          font-weight: 600;
          color: #212529;
          line-height: 1.4;
          margin-bottom: 12px;
        }
        
        .price-badge {
          background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
          color: white;
          font-size: 1rem;
          font-weight: 600;
          padding: 8px 16px;
          border-radius: 20px;
          border: none;
          box-shadow: 0 2px 10px rgba(40, 167, 69, 0.3);
        }
        
        .product-description {
          color: #6c757d;
          font-size: 0.95rem;
          line-height: 1.5;
          margin-bottom: 16px;
        }
        
        .features-section {
          margin-top: 16px;
        }
        
        .features-title {
          font-size: 1rem;
          font-weight: 600;
          color: #495057;
          margin-bottom: 12px;
          display: flex;
          align-items: center;
        }
        
        .features-icon {
          margin-right: 8px;
          color: #0d6efd;
        }
        
        .feature-item {
          border: none;
          padding: 10px 0;
          background: transparent;
          transition: all 0.3s ease;
          border-radius: 6px;
          margin: 2px 0;
        }
        
        .feature-item:hover {
          background: #f8f9fa;
          transform: translateX(4px);
        }
        
        .feature-icon {
          color: #28a745;
          margin-right: 8px;
          font-size: 0.9rem;
        }
        
        .feature-text {
          color: #495057;
          font-size: 0.9rem;
          line-height: 1.4;
        }
        
        .more-features {
          color: #6c757d;
          font-style: italic;
          font-size: 0.85rem;
        }
        
        .shimmer {
          background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
          background-size: 200% 100%;
          animation: shimmer 1.5s infinite;
        }
        
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
      `}</style>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="preview-card">
          <div className="card-header-custom">
            <h5 className="header-title">
              <i className="bi bi-eye header-icon"></i>
              Product Preview
            </h5>
          </div>
          
          <div className="product-image-container">
            {imageLoading && (
              <div className="shimmer" style={{ height: '100%' }}></div>
            )}
            {!imageError && product.images?.[0] ? (
              <Image
                src={product.images[0]}
                alt={product.title}
                className={`product-image ${imageLoading ? 'd-none' : ''}`}
                onLoad={handleImageLoad}
                onError={handleImageError}
              />
            ) : (
              !imageLoading && (
                <div className="image-placeholder">
                  <i className="bi bi-image placeholder-icon"></i>
                  <small>No image available</small>
                </div>
              )
            )}
          </div>
          
          <Card.Body className="p-3">
            <h4 className="product-title">{product.title}</h4>
            
            <div className="text-center mb-3">
              <Badge className="price-badge">{product.price}</Badge>
            </div>
            
            {product.description && (
              <p className="product-description">
                {product.description.length > 150 
                  ? `${product.description.substring(0, 150)}...` 
                  : product.description
                }
              </p>
            )}
            
            {product.features && product.features.length > 0 && (
              <div className="features-section">
                <h6 className="features-title">
                  <i className="bi bi-check-circle features-icon"></i>
                  Key Features
                </h6>
                <ListGroup variant="flush">
                  {product.features.slice(0, 4).map((feature, idx) => (
                    <ListGroup.Item key={idx} className="feature-item px-0">
                      <i className="bi bi-check feature-icon"></i>
                      <span className="feature-text">{feature}</span>
                    </ListGroup.Item>
                  ))}
                  {product.features.length > 4 && (
                    <ListGroup.Item className="feature-item px-0">
                      <i className="bi bi-plus-circle feature-icon"></i>
                      <span className="more-features">
                        +{product.features.length - 4} more features
                      </span>
                    </ListGroup.Item>
                  )}
                </ListGroup>
              </div>
            )}
          </Card.Body>
        </Card>
      </motion.div>
    </>
  );
}

