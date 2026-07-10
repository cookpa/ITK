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

#include "itkInverseDisplacementFieldImageFilter.h"
#include "itkThinPlateSplineKernelTransform.h"
#include "itkImageRegionConstIterator.h"
#include "itkGTest.h"

namespace
{
using VectorType = itk::Vector<float, 2>;
using FieldType = itk::Image<VectorType, 2>;
using FilterType = itk::InverseDisplacementFieldImageFilter<FieldType, FieldType>;
} // namespace

// The kernel-spline subsampler must inherit the input direction; the exact inverse of a
// constant displacement is its negation, reproduced exactly by the spline's affine part
// (issue #6575, B32).
TEST(InverseDisplacementFieldImageFilter, RecoversConstantInverseOnRotatedField)
{
  auto field = FieldType::New();
  field->SetRegions(FieldType::RegionType{ FieldType::IndexType{}, FieldType::SizeType::Filled(32) });
  FieldType::DirectionType direction;
  direction(0, 0) = 0.0;
  direction(0, 1) = -1.0;
  direction(1, 0) = 1.0;
  direction(1, 1) = 0.0;
  field->SetDirection(direction);
  field->Allocate();
  field->FillBuffer(itk::MakeVector(2.0F, 3.0F));

  auto filter = FilterType::New();
  filter->SetInput(field);
  filter->SetKernelTransform(itk::ThinPlateSplineKernelTransform<double, 2>::New());
  filter->SetSubsamplingFactor(4);
  filter->SetSize(FieldType::SizeType::Filled(16));
  filter->SetOutputSpacing(itk::MakeFilled<FieldType::SpacingType>(1.0));
  // Output window inside the subsampled landmark hull, so no sample extrapolates.
  filter->SetOutputOrigin(itk::MakePoint(-16.0, 8.0));
  filter->UpdateLargestPossibleRegion();

  const FieldType *                        output = filter->GetOutput();
  itk::ImageRegionConstIterator<FieldType> it(output, output->GetBufferedRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    EXPECT_NEAR(it.Get()[0], -2.0, 0.05);
    EXPECT_NEAR(it.Get()[1], -3.0, 0.05);
  }
}
