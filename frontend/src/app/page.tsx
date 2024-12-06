"use client";

import { useState, useEffect } from "react";

interface Prediction {
    bbox: number[];
    confidence: number;
    animal_type: string;
    class_confidence: number;
}

interface ApiResponse {
    status: string;
    predictions: Prediction[];
    processing_time_ms: number;
}

export default function Home() {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [loading, setLoading] = useState(false);

    const handleImageUpload = async (
        e: React.ChangeEvent<HTMLInputElement>
    ) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Hiển thị preview ảnh
        const reader = new FileReader();
        reader.onload = (e) => {
            setSelectedImage(e.target?.result as string);
        };
        reader.readAsDataURL(file);

        // Gửi ảnh lên API
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData,
            });
            const data: ApiResponse = await response.json();
            setPredictions(data.predictions);
        } catch (error) {
            console.error("Error:", error);
            setPredictions([]);
        }
        setLoading(false);
    };

    const drawBoundingBoxes = (predictions: Prediction[]) => {
        const image = document.getElementById(
            "preview-image"
        ) as HTMLImageElement;
        const canvas = document.getElementById(
            "bbox-canvas"
        ) as HTMLCanvasElement;
        if (!image || !canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Set canvas size to match original image dimensions
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;

        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        predictions.forEach((pred) => {
            const [x, y, width, height] = pred.bbox;

            // Draw rectangle
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width - x, height - y);

            // Draw label background
            ctx.fillStyle = "rgba(0, 255, 0, 0.7)";
            const label = `${pred.animal_type} ${(
                pred.class_confidence * 100
            ).toFixed(0)}%`;
            ctx.font = "24px Arial";
            const labelWidth = ctx.measureText(label).width + 10;
            ctx.fillRect(x, y - 35, labelWidth, 30);

            // Draw label text
            ctx.fillStyle = "#000000";
            ctx.fillText(label, x + 5, y - 10);
        });
    };

    // Thêm useEffect để vẽ bbox khi predictions thay đổi
    useEffect(() => {
        if (predictions.length > 0) {
            drawBoundingBoxes(predictions);
        }
    }, [predictions]);

    return (
        <main className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-6xl mx-auto">
                {/* Header với logo */}
                <div className="flex items-center justify-center mb-8">
                    <div className="text-center">
                        <img
                            src="/logo-1.png"
                            alt="EAUT Logo"
                            className="h-24 mx-auto mb-4"
                        />
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow-xl p-8 mb-12">
                    <h1 className="text-3xl font-bold text-gray-900 mb-6">
                        Hệ Thống Nhận Dạng và Phân Loại Động Vật
                    </h1>

                    <div className="prose prose-lg">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">
                            Về Dự Án
                        </h2>
                        <p className="text-gray-600 mb-6">
                            Dự án này tập trung vào việc xây dựng một hệ thống
                            trí tuệ nhân tạo có khả năng nhận dạng và phân loại
                            động vật thành hai nhóm chính: động vật ăn thịt
                            (carnivore) và động vật ăn cỏ (herbivore) từ hình
                            ảnh. Hệ thống sử dụng các kỹ thuật deep learning
                            tiên tiến để đạt độ chính xác cao trong việc phân
                            loại.
                        </p>

                        <h2 className="text-xl font-semibold text-gray-800 mb-4">
                            Nhóm Thực Hiện
                        </h2>
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div>
                                <h3 className="font-medium text-gray-700">
                                    Giảng viên hướng dẫn:
                                </h3>
                                <p className="text-gray-600">
                                    Lương Thị Hồng Lan
                                </p>
                            </div>
                            <div>
                                <h3 className="font-medium text-gray-700">
                                    Sinh viên thực hiện:
                                </h3>
                                <ul className="list-disc list-inside text-gray-600">
                                    <li>Vũ Quang Huy - 20210611</li>
                                    <li>Lò Tiến Anh - 20210526</li>
                                </ul>
                            </div>
                        </div>

                        <div className="bg-blue-50 rounded-lg p-4">
                            <h3 className="font-medium text-blue-800 mb-2">
                                Tính năng chính:
                            </h3>
                            <ul className="list-disc list-inside text-blue-700 space-y-1">
                                <li>
                                    Nhận dạng động vật trong ảnh với độ chính
                                    xác cao
                                </li>
                                <li>
                                    Phân loại thành 2 nhóm: động vật ăn thịt và
                                    ăn cỏ
                                </li>
                                <li>
                                    Hiển thị vị trí của động vật trong ảnh
                                    (bounding box)
                                </li>
                                <li>
                                    Cung cấp độ tin cậy của kết quả phân loại
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div className="text-center">
                    <h2 className="text-4xl font-bold text-gray-900 mb-8">
                        Demo Hệ Thống
                    </h2>
                    <p className="text-lg text-gray-600 mb-12">
                        Tải lên hình ảnh động vật và để AI phân loại giúp bạn
                    </p>
                </div>

                <div className="bg-white rounded-lg shadow-xl p-8">
                    <div className="space-y-8">
                        {/* Upload Section */}
                        <div className="flex justify-center">
                            <label className="relative cursor-pointer">
                                <div
                                    className="relative"
                                    style={{
                                        maxWidth: "100%",
                                        overflow: "auto",
                                    }}
                                >
                                    {selectedImage ? (
                                        <>
                                            <img
                                                id="preview-image"
                                                src={selectedImage}
                                                alt="Preview"
                                                className="max-w-none"
                                                style={{ display: "block" }}
                                                onLoad={() =>
                                                    predictions.length > 0 &&
                                                    drawBoundingBoxes(
                                                        predictions
                                                    )
                                                }
                                            />
                                            <canvas
                                                id="bbox-canvas"
                                                className="absolute top-0 left-0 pointer-events-none"
                                                style={{
                                                    width: "100%",
                                                    height: "100%",
                                                }}
                                            />
                                        </>
                                    ) : (
                                        <div className="w-[640px] h-[480px] border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center">
                                            <p className="text-sm text-gray-500">
                                                Click để chọn ảnh
                                            </p>
                                        </div>
                                    )}
                                </div>
                                <input
                                    type="file"
                                    className="hidden"
                                    accept="image/*"
                                    onChange={handleImageUpload}
                                />
                            </label>
                        </div>

                        {/* Result Section */}
                        {loading && (
                            <div className="text-center">
                                <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-primary border-t-transparent"></div>
                                <p className="mt-2 text-gray-600">
                                    Đang phân tích...
                                </p>
                            </div>
                        )}

                        {predictions.length > 0 && !loading && (
                            <div className="mt-8 bg-white rounded-lg shadow-xl p-8">
                                <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                                    Kết quả phân tích
                                </h2>
                                <div className="space-y-4">
                                    {predictions.map((pred, index) => (
                                        <div
                                            key={index}
                                            className="border rounded-lg p-4 bg-gray-50"
                                        >
                                            <div className="grid grid-cols-2 gap-4">
                                                <div>
                                                    <p className="text-sm font-medium text-gray-500">
                                                        Loại động vật
                                                    </p>
                                                    <p className="text-lg text-primary capitalize">
                                                        {pred.animal_type}
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className="text-sm font-medium text-gray-500">
                                                        Độ tin cậy phát hiện
                                                    </p>
                                                    <p className="text-lg text-primary">
                                                        {(
                                                            pred.confidence *
                                                            100
                                                        ).toFixed(2)}
                                                        %
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className="text-sm font-medium text-gray-500">
                                                        Độ tin cậy phân loại
                                                    </p>
                                                    <p className="text-lg text-primary">
                                                        {(
                                                            pred.class_confidence *
                                                            100
                                                        ).toFixed(2)}
                                                        %
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className="text-sm font-medium text-gray-500">
                                                        Vị trí (bbox)
                                                    </p>
                                                    <p className="text-sm text-gray-600">
                                                        [
                                                        {pred.bbox
                                                            .map((n) =>
                                                                n.toFixed(0)
                                                            )
                                                            .join(", ")}
                                                        ]
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Instructions */}
                <div className="mt-12 bg-white rounded-lg shadow-lg p-6">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">
                        Hướng dẫn sử dụng
                    </h2>
                    <ol className="list-decimal list-inside space-y-2 text-gray-600">
                        <li>Click vào ô upload để chọn ảnh từ máy tính</li>
                        <li>Đợi trong giây lát để hệ thống phân tích</li>
                        <li>Kết quả sẽ hiển thị ngay sau khi phân tích xong</li>
                    </ol>
                </div>
            </div>
        </main>
    );
}
