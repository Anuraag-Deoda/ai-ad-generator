import { useState, useEffect } from 'react';
import { Card, Form, Row, Col, Badge, ButtonGroup, Button } from 'react-bootstrap';

type PricingConfig = {
  enabled: boolean;
  original: string;
  sale: string;
  animation: string;
};

type CountdownConfig = {
  enabled: boolean;
  hours: number;
  style: string;
};

type RatingConfig = {
  enabled: boolean;
  value: number;
  count: number | null;
};

type ReviewConfig = {
  enabled: boolean;
  quote: string;
  author: string;
};

type CTAConfig = {
  text: string;
  style: string;
  color: number[];
};

export type DynamicContentConfig = {
  pricing: PricingConfig;
  countdown: CountdownConfig;
  rating: RatingConfig;
  review: ReviewConfig;
  cta: CTAConfig;
};

type Props = {
  config: DynamicContentConfig;
  onChange: (config: DynamicContentConfig) => void;
};

const PRICE_ANIMATIONS = [
  { value: 'drop', label: 'Drop In' },
  { value: 'slide', label: 'Slide' },
  { value: 'flash', label: 'Flash' },
  { value: 'bounce', label: 'Bounce' },
];

const COUNTDOWN_STYLES = [
  { value: 'flip', label: 'Flip Clock' },
  { value: 'digital', label: 'Digital' },
  { value: 'minimal', label: 'Minimal' },
  { value: 'urgent', label: 'Urgent' },
];

const CTA_STYLES = [
  { value: 'pulse', label: 'Pulse', icon: 'bi-heart-pulse' },
  { value: 'shake', label: 'Shake', icon: 'bi-phone-vibrate' },
  { value: 'glow', label: 'Glow', icon: 'bi-brightness-high' },
  { value: 'slide', label: 'Slide', icon: 'bi-arrow-right-circle' },
];

const CTA_PRESETS = [
  { text: 'Shop Now', color: [255, 87, 51] },
  { text: 'Buy Now', color: [0, 123, 255] },
  { text: 'Get Yours', color: [40, 167, 69] },
  { text: 'Order Today', color: [255, 193, 7] },
  { text: 'Learn More', color: [108, 117, 125] },
];

export default function DynamicContentOptions({ config, onChange }: Props) {
  const updatePricing = (field: keyof PricingConfig, value: string | boolean) => {
    onChange({
      ...config,
      pricing: { ...config.pricing, [field]: value },
    });
  };

  const updateCountdown = (field: keyof CountdownConfig, value: string | number | boolean) => {
    onChange({
      ...config,
      countdown: { ...config.countdown, [field]: value },
    });
  };

  const updateRating = (field: keyof RatingConfig, value: number | boolean | null) => {
    onChange({
      ...config,
      rating: { ...config.rating, [field]: value },
    });
  };

  const updateReview = (field: keyof ReviewConfig, value: string | boolean) => {
    onChange({
      ...config,
      review: { ...config.review, [field]: value },
    });
  };

  const updateCTA = (field: keyof CTAConfig, value: string | number[]) => {
    onChange({
      ...config,
      cta: { ...config.cta, [field]: value },
    });
  };

  const rgbToHex = (rgb: number[]): string => {
    return '#' + rgb.map(x => x.toString(16).padStart(2, '0')).join('');
  };

  const hexToRgb = (hex: string): number[] => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
      parseInt(result[1], 16),
      parseInt(result[2], 16),
      parseInt(result[3], 16)
    ] : [255, 87, 51];
  };

  return (
    <Card className="mb-3 border-0 shadow-sm">
      <Card.Header className="bg-white border-bottom">
        <div className="d-flex align-items-center">
          <i className="bi bi-stars me-2 text-warning"></i>
          <span className="fw-semibold">Dynamic Content</span>
          <Badge bg="success" className="ms-2">Pro</Badge>
        </div>
      </Card.Header>
      <Card.Body>
        {/* Pricing Display */}
        <div className="mb-4">
          <div className="d-flex align-items-center mb-2">
            <Form.Check
              type="switch"
              id="pricing-toggle"
              checked={config.pricing.enabled}
              onChange={(e) => updatePricing('enabled', e.target.checked)}
              className="me-2"
            />
            <span className="fw-medium">
              <i className="bi bi-tags me-2 text-success"></i>
              Animated Pricing
            </span>
          </div>
          {config.pricing.enabled && (
            <div className="ps-4 border-start border-2 border-success">
              <Row className="mb-2">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="small text-muted">Original Price</Form.Label>
                    <Form.Control
                      type="text"
                      placeholder="$99.99"
                      value={config.pricing.original}
                      onChange={(e) => updatePricing('original', e.target.value)}
                      size="sm"
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="small text-muted">Sale Price</Form.Label>
                    <Form.Control
                      type="text"
                      placeholder="$49.99"
                      value={config.pricing.sale}
                      onChange={(e) => updatePricing('sale', e.target.value)}
                      size="sm"
                    />
                  </Form.Group>
                </Col>
              </Row>
              <Form.Group>
                <Form.Label className="small text-muted">Animation Style</Form.Label>
                <div className="d-flex flex-wrap gap-2">
                  {PRICE_ANIMATIONS.map((anim) => (
                    <Button
                      key={anim.value}
                      variant={config.pricing.animation === anim.value ? 'success' : 'outline-secondary'}
                      size="sm"
                      onClick={() => updatePricing('animation', anim.value)}
                    >
                      {anim.label}
                    </Button>
                  ))}
                </div>
              </Form.Group>
              {config.pricing.original && config.pricing.sale && (
                <div className="mt-2 p-2 bg-dark text-white rounded text-center">
                  <span className="text-decoration-line-through text-muted me-2">
                    {config.pricing.original}
                  </span>
                  <span className="text-success fw-bold fs-5">{config.pricing.sale}</span>
                  {(() => {
                    const orig = parseFloat(config.pricing.original.replace(/[^0-9.]/g, ''));
                    const sale = parseFloat(config.pricing.sale.replace(/[^0-9.]/g, ''));
                    if (!isNaN(orig) && !isNaN(sale) && orig > 0) {
                      const discount = Math.round((1 - sale / orig) * 100);
                      return <Badge bg="danger" className="ms-2">{discount}% OFF</Badge>;
                    }
                    return null;
                  })()}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Countdown Timer */}
        <div className="mb-4">
          <div className="d-flex align-items-center mb-2">
            <Form.Check
              type="switch"
              id="countdown-toggle"
              checked={config.countdown.enabled}
              onChange={(e) => updateCountdown('enabled', e.target.checked)}
              className="me-2"
            />
            <span className="fw-medium">
              <i className="bi bi-clock-history me-2 text-danger"></i>
              Countdown Timer
            </span>
          </div>
          {config.countdown.enabled && (
            <div className="ps-4 border-start border-2 border-danger">
              <Row className="align-items-end mb-2">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="small text-muted">Duration (hours)</Form.Label>
                    <Form.Control
                      type="number"
                      min={1}
                      max={168}
                      value={config.countdown.hours}
                      onChange={(e) => updateCountdown('hours', parseInt(e.target.value) || 24)}
                      size="sm"
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="small text-muted">Style</Form.Label>
                    <Form.Select
                      value={config.countdown.style}
                      onChange={(e) => updateCountdown('style', e.target.value)}
                      size="sm"
                    >
                      {COUNTDOWN_STYLES.map((style) => (
                        <option key={style.value} value={style.value}>
                          {style.label}
                        </option>
                      ))}
                    </Form.Select>
                  </Form.Group>
                </Col>
              </Row>
              <div className="p-2 bg-dark text-white rounded text-center">
                <small className="text-muted d-block">OFFER ENDS IN</small>
                <span className="font-monospace fs-5">
                  {Math.floor(config.countdown.hours / 24).toString().padStart(2, '0')}:
                  {(config.countdown.hours % 24).toString().padStart(2, '0')}:00:00
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Star Rating */}
        <div className="mb-4">
          <div className="d-flex align-items-center mb-2">
            <Form.Check
              type="switch"
              id="rating-toggle"
              checked={config.rating.enabled}
              onChange={(e) => updateRating('enabled', e.target.checked)}
              className="me-2"
            />
            <span className="fw-medium">
              <i className="bi bi-star-fill me-2 text-warning"></i>
              Star Rating
            </span>
          </div>
          {config.rating.enabled && (
            <div className="ps-4 border-start border-2 border-warning">
              <Row className="align-items-end mb-2">
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="small text-muted">
                      Rating: {config.rating.value.toFixed(1)}
                    </Form.Label>
                    <Form.Range
                      min={0}
                      max={50}
                      value={config.rating.value * 10}
                      onChange={(e) => updateRating('value', parseInt(e.target.value) / 10)}
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group>
                    <Form.Label className="small text-muted">Review Count</Form.Label>
                    <Form.Control
                      type="number"
                      min={0}
                      placeholder="1,250"
                      value={config.rating.count || ''}
                      onChange={(e) => updateRating('count', e.target.value ? parseInt(e.target.value) : null)}
                      size="sm"
                    />
                  </Form.Group>
                </Col>
              </Row>
              <div className="p-2 bg-dark text-white rounded text-center">
                <span className="text-warning">
                  {[...Array(5)].map((_, i) => (
                    <i
                      key={i}
                      className={`bi ${i < Math.floor(config.rating.value) ? 'bi-star-fill' : i < config.rating.value ? 'bi-star-half' : 'bi-star'}`}
                    ></i>
                  ))}
                </span>
                <span className="ms-2">
                  {config.rating.value.toFixed(1)}
                  {config.rating.count && (
                    <small className="text-muted ms-1">({config.rating.count.toLocaleString()} reviews)</small>
                  )}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Review Quote */}
        <div className="mb-4">
          <div className="d-flex align-items-center mb-2">
            <Form.Check
              type="switch"
              id="review-toggle"
              checked={config.review.enabled}
              onChange={(e) => updateReview('enabled', e.target.checked)}
              className="me-2"
            />
            <span className="fw-medium">
              <i className="bi bi-chat-quote me-2 text-info"></i>
              Review Quote
            </span>
          </div>
          {config.review.enabled && (
            <div className="ps-4 border-start border-2 border-info">
              <Form.Group className="mb-2">
                <Form.Label className="small text-muted">Quote</Form.Label>
                <Form.Control
                  as="textarea"
                  rows={2}
                  placeholder="This product changed my life! Highly recommend."
                  value={config.review.quote}
                  onChange={(e) => updateReview('quote', e.target.value)}
                  size="sm"
                />
              </Form.Group>
              <Form.Group className="mb-2">
                <Form.Label className="small text-muted">Author</Form.Label>
                <Form.Control
                  type="text"
                  placeholder="Sarah M., Verified Buyer"
                  value={config.review.author}
                  onChange={(e) => updateReview('author', e.target.value)}
                  size="sm"
                />
              </Form.Group>
              {config.review.quote && (
                <div className="p-3 bg-dark text-white rounded">
                  <i className="bi bi-quote fs-4 text-info"></i>
                  <p className="mb-1 fst-italic">{config.review.quote}</p>
                  {config.review.author && (
                    <small className="text-muted">- {config.review.author}</small>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* CTA Button */}
        <div>
          <div className="d-flex align-items-center mb-2">
            <i className="bi bi-cursor-fill me-2 text-primary"></i>
            <span className="fw-medium">Call-to-Action Button</span>
          </div>
          <div className="ps-4 border-start border-2 border-primary">
            <Form.Group className="mb-2">
              <Form.Label className="small text-muted">Button Text</Form.Label>
              <div className="d-flex flex-wrap gap-2 mb-2">
                {CTA_PRESETS.map((preset) => (
                  <Button
                    key={preset.text}
                    variant={config.cta.text === preset.text ? 'primary' : 'outline-secondary'}
                    size="sm"
                    onClick={() => {
                      updateCTA('text', preset.text);
                      updateCTA('color', preset.color);
                    }}
                  >
                    {preset.text}
                  </Button>
                ))}
              </div>
              <Form.Control
                type="text"
                placeholder="Custom CTA text"
                value={config.cta.text}
                onChange={(e) => updateCTA('text', e.target.value)}
                size="sm"
              />
            </Form.Group>

            <Form.Group className="mb-2">
              <Form.Label className="small text-muted">Animation Style</Form.Label>
              <div className="d-flex flex-wrap gap-2">
                {CTA_STYLES.map((style) => (
                  <Button
                    key={style.value}
                    variant={config.cta.style === style.value ? 'primary' : 'outline-secondary'}
                    size="sm"
                    onClick={() => updateCTA('style', style.value)}
                  >
                    <i className={`${style.icon} me-1`}></i>
                    {style.label}
                  </Button>
                ))}
              </div>
            </Form.Group>

            <Row className="align-items-end">
              <Col md={6}>
                <Form.Group>
                  <Form.Label className="small text-muted">Button Color</Form.Label>
                  <div className="d-flex align-items-center gap-2">
                    <Form.Control
                      type="color"
                      value={rgbToHex(config.cta.color)}
                      onChange={(e) => updateCTA('color', hexToRgb(e.target.value))}
                      style={{ width: 50, height: 38 }}
                    />
                    <Form.Control
                      type="text"
                      value={rgbToHex(config.cta.color)}
                      onChange={(e) => updateCTA('color', hexToRgb(e.target.value))}
                      size="sm"
                    />
                  </div>
                </Form.Group>
              </Col>
              <Col md={6}>
                <div
                  className="p-2 rounded text-center text-white fw-bold"
                  style={{
                    backgroundColor: rgbToHex(config.cta.color),
                    animation: config.cta.style === 'pulse' ? 'pulse 1.5s infinite' : undefined,
                  }}
                >
                  {config.cta.text || 'Shop Now'}
                </div>
              </Col>
            </Row>
          </div>
        </div>
      </Card.Body>

      <style>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.05); opacity: 0.9; }
        }
      `}</style>
    </Card>
  );
}
