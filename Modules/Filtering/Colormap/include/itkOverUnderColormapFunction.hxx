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
#ifndef itkOverUnderColormapFunction_hxx
#define itkOverUnderColormapFunction_hxx

namespace itk::Function
{
template <typename TScalar, typename TRGBPixel>
auto
OverUnderColormapFunction<TScalar, TRGBPixel>::operator()(const TScalar & v) const -> RGBPixelType
{
  // Map the input scalar between [0, 1].
  const RealType value = this->RescaleInputValue(v);

  // Apply the color mapping.
  RealType red = value;
  RealType green = value;
  RealType blue = value;

  if (value == 0.0)
  {
    // pixel is saturated in the dark
    red = 0.0;
    green = 0.0;
    blue = 1.0;
  }
  else if (value == 1.0)
  {
    // pixel is saturated in the white
    red = 1.0;
    green = 0.0;
    blue = 0.0;
  }

  // Set the rgb components after rescaling the values.
  RGBPixelType pixel;
  NumericTraits<TRGBPixel>::SetLength(pixel, 3);

  pixel[0] = this->RescaleRGBComponentValue(red);
  pixel[1] = this->RescaleRGBComponentValue(green);
  pixel[2] = this->RescaleRGBComponentValue(blue);

  return pixel;
}
} // namespace itk::Function

#endif
