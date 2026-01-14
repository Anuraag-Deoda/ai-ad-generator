import { useState, useEffect } from 'react';
import { Card, Form, Row, Col, Badge, Button, Spinner } from 'react-bootstrap';

type Platform = {
  id: string;
  name: string;
  aspect_ratio: string;
  resolution: string;
  max_duration: number | null;
  recommended_duration: number;
};

type Props = {
  selectedPlatforms: string[];
  onChange: (platforms: string[]) => void;
};

const PLATFORM_ICONS: Record<string, string> = {
  tiktok: 'bi-tiktok',
  instagram_feed: 'bi-instagram',
  instagram_story: 'bi-instagram',
  instagram_reels: 'bi-instagram',
  youtube_short: 'bi-youtube',
  youtube_standard: 'bi-youtube',
  facebook_feed: 'bi-facebook',
  facebook_story: 'bi-facebook',
  linkedin: 'bi-linkedin',
  twitter: 'bi-twitter-x',
};

const PLATFORM_COLORS: Record<string, string> = {
  tiktok: '#000000',
  instagram_feed: '#E1306C',
  instagram_story: '#E1306C',
  instagram_reels: '#E1306C',
  youtube_short: '#FF0000',
  youtube_standard: '#FF0000',
  facebook_feed: '#1877F2',
  facebook_story: '#1877F2',
  linkedin: '#0A66C2',
  twitter: '#000000',
};

export default function PlatformSelector({ selectedPlatforms, onChange }: Props) {
  const [platforms, setPlatforms] = useState<Platform[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:5000/api/video-platforms')
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setPlatforms(data.platforms);
        }
      })
      .catch((err) => {
        console.error('Failed to load platforms:', err);
        // Fallback platforms
        setPlatforms([
          { id: 'tiktok', name: 'TikTok', aspect_ratio: '9:16', resolution: '1080x1920', max_duration: 180, recommended_duration: 30 },
          { id: 'instagram_feed', name: 'Instagram Feed', aspect_ratio: '1:1', resolution: '1080x1080', max_duration: 60, recommended_duration: 30 },
          { id: 'youtube_short', name: 'YouTube Short', aspect_ratio: '9:16', resolution: '1080x1920', max_duration: 60, recommended_duration: 30 },
          { id: 'linkedin', name: 'LinkedIn', aspect_ratio: '1:1', resolution: '1080x1080', max_duration: 600, recommended_duration: 30 },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  const togglePlatform = (platformId: string) => {
    if (selectedPlatforms.includes(platformId)) {
      onChange(selectedPlatforms.filter((p) => p !== platformId));
    } else {
      onChange([...selectedPlatforms, platformId]);
    }
  };

  const selectAll = () => {
    onChange(platforms.map((p) => p.id));
  };

  const clearAll = () => {
    onChange([]);
  };

  if (loading) {
    return (
      <Card className="mb-3 border-0 shadow-sm">
        <Card.Body className="text-center py-4">
          <Spinner animation="border" size="sm" className="me-2" />
          Loading platforms...
        </Card.Body>
      </Card>
    );
  }

  return (
    <Card className="mb-3 border-0 shadow-sm">
      <Card.Header className="bg-white border-bottom">
        <div className="d-flex align-items-center justify-content-between">
          <div className="d-flex align-items-center">
            <i className="bi bi-share me-2 text-primary"></i>
            <span className="fw-semibold">Export Platforms</span>
            {selectedPlatforms.length > 0 && (
              <Badge bg="primary" className="ms-2">
                {selectedPlatforms.length} selected
              </Badge>
            )}
          </div>
          <div className="d-flex gap-2">
            <Button variant="outline-secondary" size="sm" onClick={selectAll}>
              Select All
            </Button>
            <Button variant="outline-secondary" size="sm" onClick={clearAll}>
              Clear
            </Button>
          </div>
        </div>
      </Card.Header>
      <Card.Body>
        <Form.Text className="text-muted d-block mb-3">
          Select platforms to generate optimized video variants for each
        </Form.Text>

        <Row className="g-2">
          {platforms.map((platform) => {
            const isSelected = selectedPlatforms.includes(platform.id);
            const icon = PLATFORM_ICONS[platform.id] || 'bi-globe';
            const color = PLATFORM_COLORS[platform.id] || '#6c757d';

            return (
              <Col key={platform.id} xs={6} md={4} lg={3}>
                <div
                  onClick={() => togglePlatform(platform.id)}
                  className={`p-3 rounded border text-center cursor-pointer ${
                    isSelected ? 'border-primary bg-primary bg-opacity-10' : 'border-light'
                  }`}
                  style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                >
                  <div className="mb-2">
                    <i
                      className={`${icon} fs-4`}
                      style={{ color: isSelected ? color : '#6c757d' }}
                    ></i>
                  </div>
                  <div className="fw-medium small">{platform.name}</div>
                  <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                    {platform.aspect_ratio} | {platform.resolution}
                  </div>
                  {isSelected && (
                    <Badge bg="primary" className="mt-2" style={{ fontSize: '0.65rem' }}>
                      <i className="bi bi-check-lg"></i>
                    </Badge>
                  )}
                </div>
              </Col>
            );
          })}
        </Row>

        {selectedPlatforms.length > 0 && (
          <div className="mt-3 p-2 bg-light rounded small">
            <i className="bi bi-info-circle me-2 text-info"></i>
            {selectedPlatforms.length} variant(s) will be generated after the main video.
            Each variant is optimized for its platform's aspect ratio and safe zones.
          </div>
        )}
      </Card.Body>
    </Card>
  );
}
