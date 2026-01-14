import { Card, Form, Row, Col, Badge } from 'react-bootstrap';

type AudioSettingsType = {
  music_style: string;
  sfx_enabled: boolean;
  beat_sync: boolean;
  music_volume: number;
  sfx_volume: number;
};

type Props = {
  settings: AudioSettingsType;
  onChange: (settings: AudioSettingsType) => void;
  currentStyle?: string;
};

const MUSIC_STYLES = [
  { value: 'auto', label: 'Auto (Match Video Style)', icon: 'bi-magic' },
  { value: 'energetic', label: 'Energetic & Upbeat', icon: 'bi-lightning-charge' },
  { value: 'professional', label: 'Professional & Corporate', icon: 'bi-briefcase' },
  { value: 'casual', label: 'Casual & Friendly', icon: 'bi-emoji-smile' },
  { value: 'luxury', label: 'Luxury & Elegant', icon: 'bi-gem' },
];

export default function AudioSettings({ settings, onChange, currentStyle }: Props) {
  const handleChange = (field: keyof AudioSettingsType, value: string | number | boolean) => {
    onChange({ ...settings, [field]: value });
  };

  return (
    <Card className="mb-3 border-0 shadow-sm">
      <Card.Header className="bg-white border-bottom">
        <div className="d-flex align-items-center justify-content-between">
          <div className="d-flex align-items-center">
            <i className="bi bi-music-note-beamed me-2 text-primary"></i>
            <span className="fw-semibold">Audio Settings</span>
          </div>
          <Badge bg="success">
            <i className="bi bi-volume-up me-1"></i>
            Audio Enabled
          </Badge>
        </div>
      </Card.Header>
      <Card.Body>
        {/* Music Style */}
        <Form.Group className="mb-3">
          <Form.Label className="fw-medium">Background Music Style</Form.Label>
          <div className="d-flex flex-wrap gap-2">
            {MUSIC_STYLES.map((style) => (
              <div
                key={style.value}
                onClick={() => handleChange('music_style', style.value)}
                className={`p-2 px-3 rounded border cursor-pointer ${
                  settings.music_style === style.value
                    ? 'border-primary bg-primary bg-opacity-10'
                    : 'border-light bg-light'
                }`}
                style={{ cursor: 'pointer' }}
              >
                <i className={`${style.icon} me-2`}></i>
                <span className="small">{style.label}</span>
              </div>
            ))}
          </div>
          {settings.music_style === 'auto' && currentStyle && (
            <Form.Text className="text-muted">
              Will use {currentStyle} music to match your video style
            </Form.Text>
          )}
        </Form.Group>

        {/* Sound Effects */}
        <Row className="mb-3">
          <Col md={6}>
            <Form.Check
              type="switch"
              id="sfx-enabled"
              label={
                <span>
                  <i className="bi bi-soundwave me-2"></i>
                  Sound Effects
                </span>
              }
              checked={settings.sfx_enabled}
              onChange={(e) => handleChange('sfx_enabled', e.target.checked)}
            />
            <Form.Text className="text-muted d-block mt-1">
              Whoosh, click, and transition sounds
            </Form.Text>
          </Col>
          <Col md={6}>
            <Form.Check
              type="switch"
              id="beat-sync"
              label={
                <span>
                  <i className="bi bi-activity me-2"></i>
                  Beat Sync
                </span>
              }
              checked={settings.beat_sync}
              onChange={(e) => handleChange('beat_sync', e.target.checked)}
            />
            <Form.Text className="text-muted d-block mt-1">
              Sync animations to music beats
            </Form.Text>
          </Col>
        </Row>

        {/* Volume Controls */}
        <Row>
          <Col md={6}>
            <Form.Group>
              <Form.Label className="small fw-medium">
                <i className="bi bi-music-note me-1"></i>
                Music Volume: {Math.round(settings.music_volume * 100)}%
              </Form.Label>
              <Form.Range
                min={0}
                max={100}
                value={settings.music_volume * 100}
                onChange={(e) => handleChange('music_volume', parseInt(e.target.value) / 100)}
              />
            </Form.Group>
          </Col>
          <Col md={6}>
            <Form.Group>
              <Form.Label className="small fw-medium">
                <i className="bi bi-volume-up me-1"></i>
                SFX Volume: {Math.round(settings.sfx_volume * 100)}%
              </Form.Label>
              <Form.Range
                min={0}
                max={100}
                value={settings.sfx_volume * 100}
                onChange={(e) => handleChange('sfx_volume', parseInt(e.target.value) / 100)}
                disabled={!settings.sfx_enabled}
              />
            </Form.Group>
          </Col>
        </Row>

        {/* Audio Preview Info */}
        <div className="mt-3 p-2 bg-light rounded small">
          <i className="bi bi-info-circle me-2 text-info"></i>
          Audio will be generated and synced with your video automatically.
          Royalty-free tracks are included.
        </div>
      </Card.Body>
    </Card>
  );
}
