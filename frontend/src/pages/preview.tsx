// frontend/src/pages/preview.tsx
import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
import VideoPlayer from '@/components/VideoPlayer';

type Script = {
  hook: string;
  pitch: string;
  features: string;
  cta: string;
};

type JobData = {
  title: string;
  script: Script;
  video_path?: string;
};

export default function PreviewPage() {
  const router = useRouter();
  const { job_id } = router.query;

  const [jobData, setJobData] = useState<JobData | null>(null);

  useEffect(() => {
  if (!job_id) return;

  fetch('http://localhost:5000/api/generate-video', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id })
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        console.log('🎬 Video rendered:', data.video_path);
      } else {
        console.error('Remotion error:', data.error);
      }
    })
    .catch(err => {
      console.error('Remotion request failed:', err);
    });
}, [job_id]);


  useEffect(() => {
    if (!job_id) return;

    fetch(`http://localhost:5000/api/job-preview/${job_id}`)
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setJobData(data.data);
        } else {
          console.error('Failed to load preview');
        }
      });
  }, [job_id]);

  if (!jobData) return <p className="text-center mt-10">Loading preview...</p>;

  return (
    <main className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-2">{jobData.title}</h1>

      <VideoPlayer jobId={job_id as string} />

      <section className="mt-6">
        <h2 className="font-semibold text-lg mb-1">Ad Script</h2>
        <p><strong>Hook:</strong> {jobData.script.hook}</p>
        <p><strong>Pitch:</strong> {jobData.script.pitch}</p>
        <p><strong>Features:</strong> {jobData.script.features}</p>
        <p><strong>CTA:</strong> {jobData.script.cta}</p>
      </section>
    </main>
  );
}
