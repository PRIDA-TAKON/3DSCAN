'use client';

import { useState } from 'react';
import { supabase } from '@/lib/supabase';
import { Upload, FileVideo, CheckCircle2, Loader2 } from 'lucide-react';

export function UploadZone({ onUploadSuccess }: { onUploadSuccess: () => void }) {
    const [isUploading, setIsUploading] = useState(false);
    const [progress, setProgress] = useState(0);

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Check file size (200MB limit)
        const MAX_SIZE = 200 * 1024 * 1024; // 200MB
        if (file.size > MAX_SIZE) {
            alert('File is too large! Maximum size allowed is 200MB.');
            return;
        }

        setIsUploading(true);
        const fileExt = file.name.split('.').pop();
        const fileName = `${Math.random().toString(36).substring(2)}.${fileExt}`;
        const filePath = `videos/${fileName}`;

        try {
            // 1. Upload to Storage with Progress Tracking (Cast to any to bypass TS error if lib version varies)
            const { data: uploadData, error: uploadError } = await (supabase.storage
                .from('3d-scans') as any)
                .upload(filePath, file, {
                    onUploadProgress: (progressEvent: any) => {
                        const percent = (progressEvent.bytesTransferred / progressEvent.totalBytes) * 100;
                        setProgress(Math.round(percent));
                    }
                });

            if (uploadError) throw uploadError;

            // 2. Get Public URL
            const { data: { publicUrl } } = supabase.storage
                .from('3d-scans')
                .getPublicUrl(filePath);

            // 3. Create DB Record and get the ID
            const { data: jobData, error: dbError } = await supabase.from('jobs').insert({
                video_url: publicUrl,
                status: 'PENDING',
                message: 'Awaiting worker...'
            }).select().single();

            if (dbError) throw dbError;

            // 4. Trigger RunPod Job
            console.log('🚀 Triggering worker for job:', jobData.id);
            const runpodRes = await fetch('/api/run-worker', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    jobId: jobData.id,
                    videoUrl: publicUrl
                })
            });

            const resData = await runpodRes.json();

            if (!runpodRes.ok) {
                console.error('❌ Worker trigger failed:', resData);
                throw new Error(resData.error || 'Failed to trigger worker');
            }

            console.log('✅ Worker triggered successfully:', resData);
            onUploadSuccess();
        } catch (error: any) {
            alert(error.message);
        } finally {
            setIsUploading(false);
            setProgress(0);
        }
    };

    return (
        <div className="glass p-8 rounded-3xl border-dashed border-2 border-white/10 flex flex-col items-center text-center">
            <div className="w-16 h-16 rounded-2xl bg-yellow-500/10 flex items-center justify-center mb-6">
                {isUploading ? <Loader2 className="animate-spin text-yellow-500" /> : <Upload className="text-yellow-500" />}
            </div>

            <h3 className="text-xl font-semibold mb-2">New 3D Scan</h3>
            <p className="text-gray-400 text-sm mb-1">Upload video file to start reconstruction</p>
            <p className="text-yellow-500/80 text-[10px] font-bold uppercase tracking-widest mb-6">Max file size: 200MB</p>

            <label className="w-full">
                <input
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleUpload}
                    disabled={isUploading}
                />
                <div className={`
          cursor-pointer py-3 px-6 rounded-xl font-medium transition-all flex items-center justify-center gap-2
          ${isUploading ? 'bg-white/5 text-gray-500 cursor-not-allowed' : 'bg-yellow-500 text-black hover:bg-yellow-400 hover:scale-[1.02] active:scale-[0.98]'}
        `}>
                    {isUploading ? (
                        <>
                            <Loader2 className="animate-spin" size={18} />
                            Uploading {progress}%
                        </>
                    ) : 'Select Video File'}
                </div>
            </label>

            {isUploading && (
                <div className="mt-4 w-full">
                    <div className="bg-white/5 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-yellow-500 h-full transition-all duration-300" style={{ width: `${progress}%` }} />
                    </div>
                </div>
            )}
        </div>
    );
}
