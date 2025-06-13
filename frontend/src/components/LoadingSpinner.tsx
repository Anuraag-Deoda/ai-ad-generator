import { Spinner, Card, ProgressBar } from 'react-bootstrap';
import { motion } from 'framer-motion';

export default function LoadingSpinner() {
  return (
    <>
      <style jsx>{`
        .loading-card {
          background: #fff;
          border: 1px solid #e9ecef;
          border-radius: 12px;
          box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
          text-align: center;
          padding: 32px 24px;
          margin: 24px 0;
        }
        
        .loading-icon {
          font-size: 3rem;
          margin-bottom: 20px;
          animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
          0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
          }
          40% {
            transform: translateY(-10px);
          }
          60% {
            transform: translateY(-5px);
          }
        }
        
        .custom-spinner {
          width: 48px;
          height: 48px;
          border: 4px solid #f8f9fa;
          border-top: 4px solid #0d6efd;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto 20px;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .loading-title {
          font-size: 1.25rem;
          font-weight: 600;
          color: #212529;
          margin-bottom: 8px;
        }
        
        .loading-subtitle {
          color: #6c757d;
          font-size: 0.95rem;
          margin-bottom: 20px;
        }
        
        .progress-container {
          max-width: 300px;
          margin: 0 auto 16px;
        }
        
        .progress-custom {
          height: 8px;
          border-radius: 4px;
          background: #f8f9fa;
          overflow: hidden;
        }
        
        .progress-bar-custom {
          background: linear-gradient(90deg, #0d6efd, #6f42c1, #0d6efd);
          background-size: 200% 100%;
          animation: progressMove 2s ease-in-out infinite;
        }
        
        @keyframes progressMove {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        
        .loading-steps {
          text-align: left;
          max-width: 280px;
          margin: 0 auto;
        }
        
        .step-item {
          display: flex;
          align-items: center;
          padding: 8px 0;
          color: #6c757d;
          font-size: 0.9rem;
        }
        
        .step-icon {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #e9ecef;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-right: 12px;
          font-size: 0.7rem;
          transition: all 0.3s ease;
        }
        
        .step-item.active .step-icon {
          background: #0d6efd;
          color: white;
        }
        
        .step-item.completed .step-icon {
          background: #28a745;
          color: white;
        }
        
        .dots-container {
          display: flex;
          justify-content: center;
          gap: 4px;
          margin-top: 16px;
        }
        
        .dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #e9ecef;
          animation: dotPulse 1.4s ease-in-out infinite both;
        }
        
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        .dot:nth-child(3) { animation-delay: 0s; }
        
        @keyframes dotPulse {
          0%, 80%, 100% {
            transform: scale(0.8);
            background: #e9ecef;
          }
          40% {
            transform: scale(1.2);
            background: #0d6efd;
          }
        }
      `}</style>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="loading-card">
          <div className="loading-icon">🔍</div>
          <div className="custom-spinner"></div>
          
          <h4 className="loading-title">Analyzing Product</h4>
          <p className="loading-subtitle">
            Our AI is extracting product details and features
          </p>
          
          <div className="progress-container">
            <div className="progress-custom">
              <div className="progress-bar-custom" style={{ width: '100%' }}></div>
            </div>
          </div>
          
          <div className="loading-steps">
            <div className="step-item completed">
              <div className="step-icon">
                <i className="bi bi-check"></i>
              </div>
              <span>Fetching product page</span>
            </div>
            <div className="step-item active">
              <div className="step-icon">
                <Spinner size="sm" />
              </div>
              <span>Extracting product data</span>
            </div>
            <div className="step-item">
              <div className="step-icon">3</div>
              <span>Processing images</span>
            </div>
            <div className="step-item">
              <div className="step-icon">4</div>
              <span>Analyzing features</span>
            </div>
          </div>
          
          <div className="dots-container">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
          
          <small className="text-muted mt-3 d-block">
            ⏱️ This usually takes 10-30 seconds
          </small>
        </Card>
      </motion.div>
    </>
  );
}

