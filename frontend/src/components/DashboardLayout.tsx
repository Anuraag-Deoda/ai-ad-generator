import React, { useState } from 'react';
import { Container, Row, Col, Nav, Navbar, Offcanvas, Button } from 'react-bootstrap';
import { motion } from 'framer-motion';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const [showSidebar, setShowSidebar] = useState(false);

  const handleCloseSidebar = () => setShowSidebar(false);
  const handleShowSidebar = () => setShowSidebar(true);

  return (
    <div className="dashboard-wrapper">
      <style jsx global>{`
        .dashboard-wrapper {
          min-height: 100vh;
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        .dashboard-navbar {
          background: #fff;
          box-shadow: 0 2px 20px rgba(0, 0, 0, 0.08);
          border-bottom: 1px solid #e9ecef;
          backdrop-filter: blur(10px);
        }
        
        .sidebar-nav {
          background: #fff;
          border-right: 1px solid #e9ecef;
          box-shadow: 2px 0 20px rgba(0, 0, 0, 0.05);
          min-height: calc(100vh - 76px);
        }
        
        .nav-link-custom {
          color: #495057;
          padding: 12px 20px;
          border-radius: 8px;
          margin: 4px 8px;
          transition: all 0.3s ease;
          display: flex;
          align-items: center;
          font-weight: 500;
        }
        
        .nav-link-custom:hover {
          background: #f8f9fa;
          color: #0d6efd;
          transform: translateX(4px);
        }
        
        .nav-link-custom.active {
          background: linear-gradient(135deg, #0d6efd 0%, #0056b3 100%);
          color: white;
          box-shadow: 0 4px 15px rgba(13, 110, 253, 0.3);
        }
        
        .nav-icon {
          margin-right: 12px;
          font-size: 1.1rem;
        }
        
        .main-content {
          padding: 24px;
          background: transparent;
        }
        
        .brand-logo {
          font-weight: 700;
          background: linear-gradient(135deg, #0d6efd 0%, #6f42c1 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          font-size: 1.5rem;
        }
        
        .mobile-toggle {
          border: none;
          background: none;
          color: #495057;
          font-size: 1.2rem;
          padding: 8px;
          border-radius: 6px;
          transition: all 0.3s ease;
        }
        
        .mobile-toggle:hover {
          background: #f8f9fa;
          color: #0d6efd;
        }
        
        .offcanvas-custom {
          background: #fff;
          border-right: 1px solid #e9ecef;
        }
        
        .stats-badge {
          background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
          color: white;
          font-size: 0.75rem;
          padding: 4px 8px;
          border-radius: 12px;
          margin-left: auto;
        }
      `}</style>

      <Container fluid className="p-0">
        <Row className="g-0">

          {/* Main Content Area */}
          <Col lg={9} xl={10}>
            <motion.div
              className="main-content"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              {children}
            </motion.div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default DashboardLayout;

