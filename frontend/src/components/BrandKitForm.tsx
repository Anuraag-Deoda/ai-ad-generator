import { useState } from 'react';
import { Card, Form, Row, Col, Button, Image, Badge } from 'react-bootstrap';

type BrandKit = {
  logo_url: string;
  primary_color: string;
  secondary_color: string;
  accent_color: string;
  logo_position: string;
  watermark_opacity: number;
};

type Props = {
  brandKit: BrandKit;
  onChange: (brandKit: BrandKit) => void;
};

const POSITION_OPTIONS = [
  { value: 'top_left', label: 'Top Left' },
  { value: 'top_right', label: 'Top Right' },
  { value: 'bottom_left', label: 'Bottom Left' },
  { value: 'bottom_right', label: 'Bottom Right' },
];

export default function BrandKitForm({ brandKit, onChange }: Props) {
  const [logoPreview, setLogoPreview] = useState<string | null>(null);
  const [logoError, setLogoError] = useState(false);

  const handleChange = (field: keyof BrandKit, value: string | number) => {
    onChange({ ...brandKit, [field]: value });
  };

  const handleLogoUrlChange = (url: string) => {
    handleChange('logo_url', url);
    setLogoError(false);
    if (url) {
      setLogoPreview(url);
    } else {
      setLogoPreview(null);
    }
  };

  return (
    <Card className="mb-3 border-0 shadow-sm">
      <Card.Header className="bg-white border-bottom">
        <div className="d-flex align-items-center">
          <i className="bi bi-palette me-2 text-primary"></i>
          <span className="fw-semibold">Brand Kit</span>
          <Badge bg="info" className="ms-2">Optional</Badge>
        </div>
      </Card.Header>
      <Card.Body>
        {/* Logo URL */}
        <Form.Group className="mb-3">
          <Form.Label className="fw-medium">Logo URL</Form.Label>
          <Form.Control
            type="url"
            placeholder="https://example.com/logo.png"
            value={brandKit.logo_url}
            onChange={(e) => handleLogoUrlChange(e.target.value)}
          />
          <Form.Text className="text-muted">
            Enter a direct URL to your logo image (PNG or SVG recommended)
          </Form.Text>
        </Form.Group>

        {/* Logo Preview */}
        {logoPreview && (
          <div className="mb-3 text-center p-3 bg-light rounded">
            <Image
              src={logoPreview}
              alt="Logo preview"
              style={{ maxHeight: 60, maxWidth: 150 }}
              onError={() => setLogoError(true)}
              onLoad={() => setLogoError(false)}
            />
            {logoError && (
              <div className="text-danger small mt-2">
                <i className="bi bi-exclamation-triangle me-1"></i>
                Could not load logo
              </div>
            )}
          </div>
        )}

        {/* Brand Colors */}
        <Row className="mb-3">
          <Col>
            <Form.Group>
              <Form.Label className="fw-medium small">Primary Color</Form.Label>
              <div className="d-flex align-items-center gap-2">
                <Form.Control
                  type="color"
                  value={brandKit.primary_color}
                  onChange={(e) => handleChange('primary_color', e.target.value)}
                  style={{ width: 50, height: 38 }}
                />
                <Form.Control
                  type="text"
                  value={brandKit.primary_color}
                  onChange={(e) => handleChange('primary_color', e.target.value)}
                  placeholder="#000000"
                  size="sm"
                />
              </div>
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label className="fw-medium small">Secondary Color</Form.Label>
              <div className="d-flex align-items-center gap-2">
                <Form.Control
                  type="color"
                  value={brandKit.secondary_color}
                  onChange={(e) => handleChange('secondary_color', e.target.value)}
                  style={{ width: 50, height: 38 }}
                />
                <Form.Control
                  type="text"
                  value={brandKit.secondary_color}
                  onChange={(e) => handleChange('secondary_color', e.target.value)}
                  placeholder="#000000"
                  size="sm"
                />
              </div>
            </Form.Group>
          </Col>
          <Col>
            <Form.Group>
              <Form.Label className="fw-medium small">Accent Color</Form.Label>
              <div className="d-flex align-items-center gap-2">
                <Form.Control
                  type="color"
                  value={brandKit.accent_color}
                  onChange={(e) => handleChange('accent_color', e.target.value)}
                  style={{ width: 50, height: 38 }}
                />
                <Form.Control
                  type="text"
                  value={brandKit.accent_color}
                  onChange={(e) => handleChange('accent_color', e.target.value)}
                  placeholder="#000000"
                  size="sm"
                />
              </div>
            </Form.Group>
          </Col>
        </Row>

        {/* Color Preview */}
        <div className="mb-3">
          <div className="small text-muted mb-2">Color Preview</div>
          <div className="d-flex rounded overflow-hidden" style={{ height: 30 }}>
            <div style={{ flex: 2, backgroundColor: brandKit.primary_color }}></div>
            <div style={{ flex: 2, backgroundColor: brandKit.secondary_color }}></div>
            <div style={{ flex: 1, backgroundColor: brandKit.accent_color }}></div>
          </div>
        </div>

        {/* Logo Position */}
        <Row className="mb-3">
          <Col md={6}>
            <Form.Group>
              <Form.Label className="fw-medium small">Logo Position</Form.Label>
              <Form.Select
                value={brandKit.logo_position}
                onChange={(e) => handleChange('logo_position', e.target.value)}
                size="sm"
              >
                {POSITION_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
          <Col md={6}>
            <Form.Group>
              <Form.Label className="fw-medium small">
                Watermark Opacity: {Math.round(brandKit.watermark_opacity * 100)}%
              </Form.Label>
              <Form.Range
                min={0}
                max={100}
                value={brandKit.watermark_opacity * 100}
                onChange={(e) => handleChange('watermark_opacity', parseInt(e.target.value) / 100)}
              />
            </Form.Group>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );
}
