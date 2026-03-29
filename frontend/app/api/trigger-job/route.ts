import { NextResponse } from 'next/server';

export async function POST(req: Request) {
    try {
        const body = await req.json();
        const { jobId, videoUrl } = body;

        if (!jobId || !videoUrl) {
            return NextResponse.json({ error: 'Missing jobId or videoUrl' }, { status: 400 });
        }

        const apiKey = process.env.RUNPOD_API_KEY;
        const endpointId = process.env.RUNPOD_ENDPOINT_ID;

        if (!apiKey || !endpointId) {
            console.error('Missing RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID environment variables');
            return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
        }

        const url = `https://api.runpod.ai/v2/${endpointId}/run`;

        const payload = {
            input: {
                id: jobId,
                video_url: videoUrl
            }
        };

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('RunPod API Error:', errorText);
            throw new Error(`RunPod API failed with status ${response.status}`);
        }

        const data = await response.json();
        return NextResponse.json({ success: true, runpodId: data.id });

    } catch (error: any) {
        console.error('Trigger job failed:', error);
        return NextResponse.json({ error: error.message || 'Internal Server Error' }, { status: 500 });
    }
}
