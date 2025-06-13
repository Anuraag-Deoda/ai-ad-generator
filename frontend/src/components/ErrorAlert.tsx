import { Alert, Button } from 'react-bootstrap';
import { motion } from 'framer-motion';

type Props = {
  message: string;
  onClose: () => void;
};

export default function ErrorAlert({ message, onClose }: Props) {
  return (
    <>
      <style jsx>{`
        .error-alert {
          background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
          border: 1px solid #f5c6cb;
          border-radius: 12px;
          color: #721c24;
          box-shadow: 0 4px 15px rgba(220, 53, 69, 0.15);
          margin-bottom: 20px;
          overflow: hidden;
          position: relative;
        }
        
        .error-alert::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 4px;
          background: linear-gradient(90deg, #dc3545, #c82333, #dc3545);
          background-size: 200% 100%;
          animation: errorPulse 2s ease-in-out infinite;
        }
        
        @keyframes errorPulse {
          0%, 100% { background-position: 200% 0; }
          50% { background-position: -200% 0; }
        }
        
        .error-content {
          padding: 16px 20px;
          position: relative;
          z-index: 1;
        }
        
        .error-header {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
        }
        
        .error-icon {
          font-size: 1.3rem;
          margin-right: 10px;
          color: #dc3545;
          animation: shake 0.5s ease-in-out;
        }
        
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-3px); }
          75% { transform: translateX(3px); }
        }
        
        .error-title {
          font-weight: 600;
          color: #721c24;
          margin: 0;
          font-size: 1.1rem;
        }
        
        .error-message {
          color: #721c24;
          margin-bottom: 12px;
          line-height: 1.5;
        }
        
        .error-actions {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 8px;
        }
        
        .error-suggestions {
          color: #856404;
          font-size: 0.85rem;
          display: flex;
          align-items: center;
          flex: 1;
        }
        
        .suggestion-icon {
          margin-right: 6px;
          color: #ffc107;
        }
        
        .close-button {
          background: rgba(114, 28, 36, 0.1);
          border: 1px solid rgba(114, 28, 36, 0.2);
          color: #721c24;
          border-radius: 6px;
          padding: 6px 12px;
          font-size: 0.85rem;
          font-weight: 500;
          transition: all 0.3s ease;
        }
        
        .close-button:hover {
          background: rgba(114, 28, 36, 0.2);
          border-color: rgba(114, 28, 36, 0.3);
          color: #721c24;
          transform: translateY(-1px);
        }
        
        .retry-button {
          background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
          border: none;
          color: white;
          border-radius: 6px;
          padding: 6px 12px;
          font-size: 0.85rem;
          font-weight: 500;
          transition: all 0.3s ease;
          margin-left: 8px;
        }
        
        .retry-button:hover {
          background: linear-gradient(135deg, #c82333 0%, #a71e2a 100%);
          transform: translateY(-1px);
          color: white;
        }
      `}</style>

      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 20 }}
        transition={{ duration: 0.4 }}
      >
        <Alert className="error-alert" dismissible={false}>
          <div className="error-content">
            <div className="error-header">
              <i className="bi bi-exclamation-triangle-fill error-icon"></i>
              <h6 className="error-title">Something went wrong</h6>
            </div>
            
            <p className="error-message">{message}</p>
            
            <div className="error-actions">
              <div className="error-suggestions">
                <i className="bi bi-lightbulb suggestion-icon"></i>
                <span>Try refreshing or check your connection</span>
              </div>
              
              <div>
                <Button 
                  className="close-button"
                  onClick={onClose}
                >
                  <i className="bi bi-x me-1"></i>
                  Dismiss
                </Button>
                <Button 
                  className="retry-button"
                  onClick={() => window.location.reload()}
                >
                  <i className="bi bi-arrow-clockwise me-1"></i>
                  Retry
                </Button>
              </div>
            </div>
          </div>
        </Alert>
      </motion.div>
    </>
  );
}

