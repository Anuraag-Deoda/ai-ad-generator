import { Spinner, Card } from 'react-bootstrap';

export default function LoadingSpinner() {
  return (
    <Card className="text-center py-5 my-4 border-0 bg-light">
      <Card.Body>
        <Spinner animation="border" variant="primary" className="mb-3" />
        <h5 className="text-muted">Analyzing product...</h5>
        <p className="text-muted mb-0">This may take a few moments</p>
      </Card.Body>
    </Card>
  );
}
