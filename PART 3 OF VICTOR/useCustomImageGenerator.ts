import { useState, useMemo } from 'react';

// Define the structure of the API responses
interface PromptResponse {
    prompt: string;
}

interface ImageResponse {
    image: string; // Base64 encoded image
}

// Define the error structure
interface ApiError {
    message: string;
}

// Define the hook's return type
interface UseCustomImageGeneratorReturn {
    generatedImage: string | null;
    loading: boolean;
    error: string | null;
    generate: (
        image1: string,
        image2: string,
        blendMode: string,
        prompt: string
    ) => Promise<void>;
}

export const useCustomImageGenerator = (): UseCustomImageGeneratorReturn => {
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const generate = async (
        image1: string,
        image2: string,
        blendMode: string,
        prompt: string
    ) => {
        setLoading(true);
        setError(null);
        setGeneratedImage(null);

        try {
            // Step 1: Generate a descriptive prompt
            const promptResponse = await fetch('/api/generate-prompt', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image1,
                    image2,
                    blendMode,
                    prompt,
                }),
            });

            if (!promptResponse.ok) {
                const errorData: ApiError = await promptResponse.json();
                throw new Error(errorData.message || 'Failed to generate prompt.');
            }

            const { prompt: descriptivePrompt }: PromptResponse = await promptResponse.json();

            // Step 2: Generate the image using the descriptive prompt
            const imageResponse = await fetch('/api/generate-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: descriptivePrompt,
                }),
            });

            if (!imageResponse.ok) {
                const errorData: ApiError = await imageResponse.json();
                throw new Error(errorData.message || 'Failed to generate image.');
            }

            const { image: base64Image }: ImageResponse = await imageResponse.json();

            if (base64Image) {
                setGeneratedImage(`data:image/jpeg;base64,${base64Image}`);
            } else {
                throw new Error('Image generation failed. The response did not contain image data.');
            }
        } catch (err) {
            console.error(err);
            setError(err instanceof Error ? err.message : 'An unknown error occurred during image generation.');
        } finally {
            setLoading(false);
        }
    };

    return { generatedImage, loading, error, generate };
};
