// frontend/src/components/VideoPlayer.tsx
type Props = {
  jobId: string;
};

export default function VideoPlayer({ jobId }: Props) {
  return (
    <div className="my-4">
      <video
        controls
        className="w-full rounded shadow"
        src={`http://localhost:5000/static/videos/${jobId}.mp4`}
      />
      <a
        href={`http://localhost:5000/static/videos/${jobId}.mp4`}
        download
        className="text-blue-600 underline block mt-2 text-center"
      >
        Download Video
      </a>
    </div>
  );
}
