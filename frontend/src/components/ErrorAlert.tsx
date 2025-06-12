import { Alert, Button } from 'react-bootstrap';

type Props = {
  message: string;
  onClose: () => void;
};

export default function ErrorAlert({ message, onClose }: Props) {
  return (
    <Alert variant="danger" dismissible onClose={onClose} className="mb-4">
      <Alert.Heading className="h6">
        <i className="bi bi-exclamation-triangle-fill me-2"></i>
        Error
      </Alert.Heading>
      <p className="mb-0">{message}</p>
    </Alert>
  );
}
