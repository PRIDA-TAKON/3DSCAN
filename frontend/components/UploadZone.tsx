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

        // Check file size (50MB limit)
        const MAX_SIZE = 50 * 1024 * 1024; // 50MB
        if (file.size > MAX_SIZE) {
            alert('File is too large! Maximum size allowed is 50MB.');
            return;
        }

        setIsUploading(true);
        const fileExt = file.name.split('.').pop();
        const fileName = `${Math.random().toString(36).substring(2)}.${fileExt}`;
        const filePath = `videos/${fileName}`;

        try {
            // 1. Upload to Storage
            const { data: uploadData, error: uploadError } = await supabase.storage
                .from('3d-scans')
                .upload(filePath, file);

            if (uploadError) throw uploadError;

            // 2. Get Public URL
            const { data: { publicUrl } } = supabase.storage
                .from('3d-scans')
                .getPublicUrl(filePath);

            // 3. Create DB Record
            const { error: dbError } = await supabase.from('jobs').insert({
                video_url: publicUrl,
                status: 'PENDING',
                message: 'Awaiting worker...'
            });

            if (dbError) throw dbError;

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
            <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-6">
                {isUploading ? <Loader2 className="animate-spin text-primary" /> : <Upload className="text-primary" />}
            </div>

            <h3 className="text-xl font-semibold mb-2">New 3D Scan</h3>
            <p className="text-gray-400 text-sm mb-6">Upload video file to start reconstruction (MP4/MOV)</p>

            <label className="w-full">
                <input
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleUpload}
                    disabled={isUploading}
                />
                <div className={`
          cursor-pointer py-3 px-6 rounded-xl font-medium transition-all
          ${isUploading ? 'bg-gray-700 opacity-50' : 'bg-primary hover:scale-[1.02] active:scale-[0.98]'}
        `}>
                    {isUploading ? 'Uploading...' : 'Select Video File'}
                </div>
            </label>

            {isUploading && (
                <div className="mt-4 w-full bg-white/5 h-2 rounded-full overflow-hidden">
                    <div className="bg-primary h-full transition-all" style={{ width: `${progress}%` }} />
                </div>
            )}
        </div>
    );
}
