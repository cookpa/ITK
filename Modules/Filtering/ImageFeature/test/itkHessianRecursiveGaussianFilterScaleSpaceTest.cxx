/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkHessianRecursiveGaussianImageFilter.h"

// This test creates an image varying as a 1D Gaussian in the X direction
// for different values of sigma, and checks the scale-space response of
// the xx component of the Hessian at the center of the Gaussian.
// If NormalizeAcrossScale works correctly, the filter should yield the
// same Hxx across different scales.

int
itkHessianRecursiveGaussianFilterScaleSpaceTest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = double;
  using ImageType = itk::Image<PixelType, Dimension>;
  using IndexType = itk::Index<Dimension>;
  using SizeType = itk::Size<Dimension>;
  using RegionType = itk::ImageRegion<Dimension>;
  using PointType = ImageType::PointType;
  using SpacingType = ImageType::SpacingType;

  auto inputImage = ImageType::New();

  auto size = SizeType::Filled(21);
  size[0] = 401;

  constexpr IndexType start{};

  RegionType region;
  region.SetIndex(start);
  region.SetSize(size);

  auto origin = itk::MakeFilled<PointType>(-1.25);
  origin[0] = -20.0;

  auto spacing = itk::MakeFilled<SpacingType>(0.1);

  inputImage->SetOrigin(origin);
  inputImage->SetSpacing(spacing);

  inputImage->SetRegions(region);
  inputImage->Allocate();

  using IteratorType = itk::ImageRegionIteratorWithIndex<ImageType>;

  constexpr unsigned int numberOfScales = 4;
  double                 scales[numberOfScales];
  scales[0] = 1.0;
  scales[1] = 2.0;
  scales[2] = 3.0;
  scales[3] = 5.0;

  // changing the size of the object with the size of the
  // gaussian should produce the same results
  for (const double objectSize : scales)
  {
    IteratorType it(inputImage, inputImage->GetRequestedRegion());

    PointType point;
    // Fill the image with a 1D Gaussian along X with sigma equal to the current scale
    // The Gaussian is not normalized, since it should have the same peak value across
    // scales, only sigma should change
    while (!it.IsAtEnd())
    {
      inputImage->TransformIndexToPhysicalPoint(it.GetIndex(), point);
      const double value = std::exp(-point[0] * point[0] / (2.0 * objectSize * objectSize));
      it.Set(value);
      ++it;
    }

    // Compute the hessian using NormalizeAcrossScale true
    using FilterType = itk::HessianRecursiveGaussianImageFilter<ImageType>;

    using HessianImageType = FilterType::OutputImageType;

    auto filter = FilterType::New();
    filter->SetInput(inputImage);
    filter->SetSigma(objectSize);
    filter->SetNormalizeAcrossScale(true);
    filter->Update();

    const HessianImageType::Pointer outputImage = filter->GetOutput();

    // Get the value at the center of the image, the location of the peak of the Gaussian
    constexpr PointType center{};

    const IndexType centerIndex = outputImage->TransformPhysicalPointToIndex(center);

    // Irrespective of the scale, the Hxx component should be the same
    const double centerHxx = outputImage->GetPixel(centerIndex)[0];

    if (centerHxx > -0.3546 || centerHxx < -0.3547)
    {
      std::cout << "center Hessian: " << outputImage->GetPixel(centerIndex) << std::endl;
      return EXIT_FAILURE;
    }
  }


  // maintaining the size of the object and gaussian, in physical
  // size, should maintain the value, while the size of the image changes.
  for (const double scale : scales)
  {
    IteratorType it(inputImage, inputImage->GetRequestedRegion());

    PointType        point;
    constexpr double objectSize = 5.0;

    spacing.Fill(scale / 5.0);

    inputImage->SetSpacing(spacing);

    // Fill the image with a 1D Gaussian along X with sigma equal to
    // the object size.
    // The Gaussian is not normalized, since it should have the same peak value across
    // scales, only sigma should change
    while (!it.IsAtEnd())
    {
      inputImage->TransformIndexToPhysicalPoint(it.GetIndex(), point);
      const double value = std::exp(-point[0] * point[0] / (2.0 * objectSize * objectSize));
      it.Set(value);
      ++it;
    }

    // Compute the hessian using NormalizeAcrossScale true
    using FilterType = itk::HessianRecursiveGaussianImageFilter<ImageType>;

    using HessianImageType = FilterType::OutputImageType;

    auto filter = FilterType::New();
    filter->SetInput(inputImage);
    filter->SetSigma(objectSize);
    filter->SetNormalizeAcrossScale(true);
    filter->Update();

    const HessianImageType::Pointer outputImage = filter->GetOutput();

    // Get the value at the center of the image, the location of the peak of the Gaussian
    constexpr PointType center{};

    const IndexType centerIndex = outputImage->TransformPhysicalPointToIndex(center);

    // Irrespective of the scale, the Hxx component should be the same
    const double centerHxx = outputImage->GetPixel(centerIndex)[0];

    if (centerHxx > -0.354 || centerHxx < -0.355)
    {
      std::cout << "center Hessian: " << outputImage->GetPixel(centerIndex) << std::endl;
      return EXIT_FAILURE;
    }
  }


  return EXIT_SUCCESS;
}
